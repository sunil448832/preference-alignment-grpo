# GRPO with Verifiable Rewards on Qwen3-4B (competition math)

RL post-training of [Qwen/Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) on
competition-level math with **GRPO** (DeepSeekMath) and a **verifiable reward**: a Python checker
(`math-verify`) decides whether the boxed answer matches the reference. No reward model, no KL reference
model. The whole pipeline (rollouts, log-probs, loss, optimizer, checkpoints, in-loop eval) is written out
in plain PyTorch in [`train_grpo_pytorch.py`](train_grpo_pytorch.py) and runs on **one 24 to 48 GB GPU**.

## Result

Paired greedy evaluation on 500 held-out problems, cap 4096 tokens, base vs. checkpoint step 275
(280 steps, 16 prompts x 8 rollouts per step, about one GPU-day on an L40S):

| | base | after GRPO |
|---|---|---|
| pass rate | 38.6% | **41.0%** (+32 solved / -20 lost, McNemar p = 0.13) |
| mean completion length | 2280 tok | 1924 tok |
| hit the 4096-token cap | 33% | 21% |
| pass rate, scoreable labels only (409) | 42.8% | 45.5% |

What the run learned is mostly *to finish*: the gain sits almost entirely in problems the base model used
to truncate. Full analysis, the baseline study that fixed the cap and prompt headroom, and the label-quality
audit (about 18% of reference answers cannot be verified as-is) are in
[`eval/BASELINE_NOTES.md`](eval/BASELINE_NOTES.md). Design decisions and a dated run log are in
[`docs/PLAN.md`](docs/PLAN.md).

## What is in the trainer

- **GRPO from scratch**: group-relative advantages (population std), clipped policy-gradient objective,
  per-sequence advantage broadcast to tokens, batch-token normalization (Dr.GRPO/DAPO convention), beta = 0.
- **Sampler/scorer consistency**: rollouts sample from a tempered softmax (T = 0.7, no top-p/top-k) and the
  scorer divides logits by the same T, so the importance ratio compares the same policy.
- **Colocated vLLM rollouts** on the training GPU: bf16 engine, the live LoRA adapter hot-swapped in every
  step via `LoRARequest`, sleep mode around the gradient steps on smaller cards.
- **QLoRA training copy**: NF4 base + LoRA r = 16 on all linear layers; chunked fp32 log-softmax under
  activation checkpointing so the 152K-vocab logits never materialise for a whole micro-batch.
- **Adaptive sampling**: a prompt whose 8 rollouts all fail is re-sampled (up to 16 rollouts, longer cap)
  until a success appears, then trained on a fixed-size group of successes plus failures. Raised the share
  of gradient-carrying groups from 17% to 42%. Exhausted and always-solved prompts are parked.
- **Token-budget micro-batching**: sequences are packed by length into micro-batches of a fixed token
  budget; failed rollouts are clipped to the longest correct rollout of the same prompt.
- **Optional length-aware reward** (`cfg.reward.use_length`): one-sided penalty on correct answers longer
  than 3x the reference solution, full at 4x. Off in the reported run.
- **Checkpoint/resume** with adapter, optimizer, scheduler, RNG and sampling state; WSD schedule so
  `max_steps` can be extended on resume.
- **In-loop eval** on a deterministic held-out subset every N steps, with per-problem flips logged.

## Layout

```
config.py                nested dataclasses: every knob (model, LoRA, batch, optim, reward, GPU profile, ckpt, eval)
train_grpo_pytorch.py    the trainer (vLLM or HF rollouts)
train_grpo_rlvr.py       same run through trl.GRPOTrainer (expected to fail on vLLM 0.28, see docstring)
reward_fn.py             math_reward (correctness) and length_reward
evaluate.py              paired before/after eval of a checkpoint on the holdout
data/                    load_and_split.py (PrimeIntellect/verifiable-math-problems -> 5000 train / 500 holdout), splits
eval/                    engines (vLLM/HF), baseline, pass@k, run analysis, results and notes
docs/PLAN.md             plan, GPU sizing, dated status log
setup_gpu.sh             driver, CUDA and venv setup for a fresh single-GPU box (tested on SageMaker g5/g6e)
```

## Setup

```bash
bash setup_gpu.sh            # fresh box: driver + venv from requirements.txt (installs vLLM first, it pins torch)
# or, with a working GPU stack:
pip install -r requirements.txt
```

Tested with vLLM 0.28.0, transformers 5.14, trl 1.13, peft 0.20, bitsandbytes 0.50 on A10G (24 GB) and
L40S (48 GB). On SageMaker images the FlashInfer sampler cannot JIT (missing CUDA headers); the code sets
`VLLM_USE_FLASHINFER_SAMPLER=0` itself.

## Run

Everything is set in [`config.py`](config.py); nothing is read from the environment.

```bash
# 1. data (5000 train / 500 holdout, competition sources only, gold_tokens column added)
python data/load_and_split.py

# 2. baseline on the holdout (greedy, vLLM bf16)
python -m eval.pilot_baseline --sample-size 200 --max-new-tokens 4096

# 3. train. First run on a new box: leave cfg.smoke = True (a few minutes, every code path),
#    then set smoke = False for the real run. GPU profile is auto-detected (24gb / 40gb / 48gb).
python train_grpo_pytorch.py 2>&1 | tee train.log

# 4. paired before/after on all 500 holdout problems
python evaluate.py checkpoints/grpo_pytorch/step_275 --max-new-tokens 4096

# 5. optional: pass@k (sampling headroom) and per-window training signal
python -m eval.passk --sample-size 200 --num-samples 8 --max-new-tokens 4096 --adapter checkpoints/grpo_pytorch/step_275
python -m eval.analyze_run checkpoints/grpo_pytorch
```

Resume is automatic: with `cfg.checkpoint.resume = "auto"` the trainer continues from the newest
`checkpoints/grpo_pytorch/step_N`.

## Data

`PrimeIntellect/verifiable-math-problems` (HF Hub), filtered to the `olympiads`, `amc_aime`,
`synthetic_amc` and `aops_forum` sources. GSM8K and MATH are excluded because Qwen decontaminates against
them. The two splits used in the reported run are committed under `data/` so results are reproducible.

## Next

Label cleaning of both splits (the largest known loss of signal), the length penalty on, pass@k on base
vs. trained to size the remaining sharpening headroom, and a curriculum over problem difficulty.
