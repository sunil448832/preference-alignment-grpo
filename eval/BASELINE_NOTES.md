# Baseline observations (Qwen3-4B-Instruct-2507, zero-shot, greedy)

Stage 1 of docs/PLAN.md. Two runs of `eval/pilot_baseline.py` on `data/eval_holdout.jsonl`, analysed with
`eval/analyze_results.py`. Reference-solution lengths come from `gold_standard_solution` in the source
dataset (PrimeIntellect/verifiable-math-problems), tokenised with the Qwen3 tokenizer.

## Runs

| run | date | rows | cap | backend | pass rate | hit cap | pass among finished | mean length |
|---|---|---|---|---|---|---|---|---|
| A | 2026-09-18 | 500 | 2000 | hf, NF4, batch 8 | 25.8% | 57.8% | 57% | 1507 |
| B | 2026-09-18 | 200 (first 200 rows) | 4096 | vLLM, bf16 | 34.5% | 34.5% | 52% | 2407 |

Run A's file is kept as `baseline_results_500_hf_cap2000.jsonl` on the GPU box; `baseline_results.jsonl`
is run B. Engine check: on the same 20 problems at cap 2000, HF scored 5/20 and vLLM 6/20, one
olympiads row differing. Greedy decoding on bf16 vs NF4 weights drifts that much; the engines agree.
vLLM does 20 problems in 50 s where HF took roughly 20 min. Run B took about 10 min for 200 rows.

Per source, run B: olympiads 36.1% (n=108, 40% truncated), synthetic_amc 34.2% (n=73, 26%),
aops_forum 38.5% (n=13, 54%), amc_aime 0.0% (n=6, 0% truncated; all label-format misses, see below).

## 1. Truncation, and why 4096 is the right cap

Run A: median completion length was exactly the cap. Over half the eval never reached an answer.
Truncated completions are not degenerate: zero of 289 showed tail repetition, and 235 of 289 contain
self-correction markers ("Wait", "re-check"). The model does long thinking-style reasoning inside a
non-thinking response.

Raising the cap to 4096 (run B) recovered ~9 points, almost all from rows that needed 2000-3000 tokens.
Only 16 rows finished between 3000 and 4096 and they passed at a low rate. The 69 rows still at 4096
have short reference solutions (median 647 tokens, max 1338): at the cap the model has already spent a
median of 6.3x the human length without an answer. Past ~3x gold, length is a symptom of being lost,
not a need for more room. Doubling again to 8192 would rescue a trickle at double the generation cost.

Pass rate by completion length, run B:

| length band | n | pass |
|---|---|---|
| 0-500 | 31 | 71% |
| 500-1000 | 27 | 44% |
| 1000-2000 | 29 | 48% |
| 2000-3000 | 28 | 54% |
| 3000-4096 | 16 | 25% |
| at cap 4096 | 69 | 1.4% |

**Decision: training and eval cap = 4096.** `train_grpo_rlvr.py` needs `max_completion_length=4096` and
`vllm_max_model_length >= 4736` (longest holdout prompt is 626 tokens; use 5120). docs/PLAN.md's GPU sizing
table assumes 2048, so the vLLM KV cache line roughly doubles. Step time roughly doubles too.

## 2. Headroom over the human solution

Reference solutions are short and efficient; the model needs more room. From run B:

| | tokens |
|---|---|
| gold, median | 588 |
| gold, p95 | 1137 |
| gold, max (200 rows / 500 rows) | 1338 / 1600 |
| correct completions, median | 1004 |
| correct completions, p95 | 3503 |
| correct completions, max | 4012 |

Model/gold length ratio on rows that finished and were correct: median 1.6x, p75 2.8x, p90 5.4x.
Wrong-but-finished rows run longer: median 2.5x. Coverage of correct rows by allowance:

| allowance | correct rows covered |
|---|---|
| 3x gold | 53 / 68 |
| 4x gold | 56 / 68 |
| 6x gold | 63 / 68 |

**Rule of thumb: allow ~3x the human solution for the typical case, and set the hard cap at ~4x the p95
gold length.** For this dataset that is 4x 1137 = ~4500, i.e. the 4096 already chosen. The cap clips
only the tail of correct answers (the longest correct completion was 4012). After GRPO the model should
get shorter on average, since only finished answers earn reward, so 4096 becomes more generous over
training rather than less.

## 3. Label quality: ~10-19% of rows cannot be scored as-is

Audit of `ground_truth` strings against what the prompt template asks for ("the number or
mathematical expression of the solution") and what `math_reward` can verify:

| gold format | run A (500) | run B (200) | pass (B) | problem |
|---|---|---|---|---|
| choice marker + value, e.g. `\textbf{(C)}\ 2^{16}` | 23 | 7 | 0% | verifier can't match letter-plus-value; model gives one or the other |
| bare letter, e.g. `C` | 43 | 20 | 25% | prompt asks for a value; model gave a value in 31/43 (A) |
| multi-part / sentence, e.g. `\begin{cases}...`, `60 or 120`, `a = b and c = 0` | 34 | 11 | 12-18% | not a single boxed answer |
| percent, `80\%` vs model's `80` | 7 | 3 | 0% | model reads the % from the question |

Run B: 38 of 200 (19%) fall in the first three rows. amc_aime's 0/6 is entirely this. Excluding them,
the base model's pass rate on scoreable rows is ~38%. The train pool has the same source mix and the
same defect rate, so these prompts produce noisy zero rewards during GRPO: the model can be right and
still get 0, and the group's advantages point in random directions.

Genuine reasoning errors among finished-but-wrong rows (after removing format misses): off-by-one
answers (22 vs 14, 2015 vs 2014, 11 vs 10), a sign error inside a square root, a wrong fraction
(643/750 vs 428/500). Roughly 14% of rows in run A, ~15% in run B.

## 4. What this means for Stage 2

Headroom is real: 34.5% baseline, in the 0.2-0.8 band, and the largest miss bucket (34.5% truncated)
is exactly what reward-on-finish fixes. At cap 4096 and temperature 1.0 roughly a third of rollouts
will still earn 0, which is workable.

Before training, in order of value:

1. **Clean labels in both splits**: drop bare-letter, choice-marker and multi-part golds; accept a bare
   number against a percent gold in `math_reward`. Problem ids stay stable, so the baseline can be
   filtered post hoc or rerun in ~10 min.
2. **Set the cap to 4096** in both training scripts (see section 1) and redo the KV-cache row of the
   GPU sizing table.
3. **Optional**: add one concision line to the prompt template and rerun the 200-row baseline. If
   truncation drops well below 34% without hurting pass rate, use that prompt for training and eval
   both; GRPO then starts with more non-zero groups. This changes the task prompt, so decide
   deliberately.

## 5. Length-aware reward (implemented 2026-09-18)

Both a 600-token and a 3000-token correct solution earn 1.0 from `math_reward`; GRPO's group-relative
advantage then has no reason to prefer the shorter one. `reward_fn.length_reward` adds that preference,
built from section 2's numbers:

- `ratio = completion_tokens / gold_tokens` (gold_tokens: reference solution in policy tokens, a column
  in both splits; `data/add_gold_tokens.py` added it to the existing files without re-splitting).
- `excess = clip((ratio - 3) / (4 - 3), 0, 1)`: no penalty up to **3x** gold, full penalty at **4x**.
- `length_reward = -excess` **only when the answer is correct**, else 0. Combined at
  `LENGTH_REWARD_WEIGHT = 0.5`: a correct answer scores 1.0 down to 0.5, a wrong one 0.

Design constraints: one-sided (shorter than gold is never penalised -- shorter and correct is strictly
better), and correctness dominates (a correct answer at any length beats every wrong one; do not raise
the weight to 1.0 or an overlong correct answer ties with a wrong one). Inside a group of several
correct completions the shortest gets the positive advantage, which is the whole point.

Switch: `config.USE_LENGTH_REWARD`. Both training scripts build
their reward list from `reward_fn.active_rewards()`, so nothing else changes when it is off.

Watch during training: eval pass rate and mean completion length together. Success = shorter at equal
or better pass rate. Pass rate falling while length falls means the model is skipping reasoning and the
weight is too high. The no-gold-length alternative (Kimi k1.5's group-relative bonus among correct
answers) is the fallback if gold lengths prove noisy.

## Eval protocol notes

- Use the same backend, quant, cap and `--limit` for before and after. `evaluate.py` pairs by
  problem_id and warns on a backend mismatch. `--limit 200` resolves ~10-point deltas, 500 resolves
  ~6-point; iterate at 200, report the final number at 500.
- `--batch-size` is irrelevant for vLLM (the engine schedules within the chunk); it only bounds
  memory on the HF path.
- vLLM needs `VLLM_USE_FLASHINFER_SAMPLER=0` on the SageMaker image (set in `eval/common.py`): the
  FlashInfer sampler JIT-compiles with nvcc and the image lacks CUDA headers. Greedy decoding doesn't
  use it anyway.

## 6. After 280 GRPO steps (2026-09-22, L40S, 16 prompts x 8, cap 4096, retries 16, length penalty off)

Paired on all 500 holdout rows, vLLM bf16 greedy, cap 4096, checkpoint step_275:

| | base | after |
|---|---|---|
| pass rate | 0.386 | **0.410** (+2.4 pts; +32 solved / -20 lost, net +12; McNemar p = 0.13) |
| mean tokens | 2280 | 1924 (0.84x) |
| at cap | 33% (163) | 21% (103) |
| finished but wrong | 144 | 192 |
| scoreable prompts only (409) | 0.428 | 0.455 |

**What the run learned: to finish.** Truncations fell by 60; 18 of the rescued problems became correct, 42 became
finished-but-wrong. The whole gain sits in the base-truncated band (1.8% -> 11.4% pass on 166 problems); every
band the base already finished is flat or slightly down (2-4K tokens: -5 to -7 pts on n = 40-44, within noise).
Per source: amc_aime +12 pts (n = 17), synthetic_amc +2.8, aops +2.6, olympiads +1.5. Label-format buckets
unchanged except choice_marker (+3 of 15). The in-loop 200-prompt curve looked flat because a 2.4-point effect is
below its +/-7 resolution; the 500-row paired eval and the monotone length/at-cap trends are the evidence.

**Reading.** A small, plausibly real gain from a behavioural change (stop rambling), no measurable accuracy gain on
problems the model already completed. Consistent with the signal-density analysis: ~18% unscoreable prompts and
17-42% gradient-carrying groups meant a few hundred useful updates. Next levers, in order: label cleaning (91/500
holdout prompts and the same share of the pool are dead weight), pass@k on base vs step_275 to size the sharpening
headroom, length penalty on, more steps at 2e-5.
