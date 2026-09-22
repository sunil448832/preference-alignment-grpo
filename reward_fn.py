"""Stage: RLVR verifier.

math_reward() is a plain Python function reward for trl.GRPOTrainer's reward_funcs list --
no reward-model weights in the loop (docs/PLAN.md, "Verifier -- math-verify").

length_reward() is a second, small reward that leans on correct-but-overlong completions. Rationale
(eval/BASELINE_NOTES.md Sec 2): correct answers sit at ~1.6x the human solution length (median), while
completions past ~3x gold are mostly lost, not thorough. The penalty is one-sided (shorter than gold is
never penalised), applies only when the answer is correct (a wrong short answer must never outscore a
right long one), and is weighted cfg.reward.weight so a correct answer scores in [1 - weight, 1] and
a wrong one 0. Inside a GRPO group that gap is enough: among several correct completions the shortest
gets the positive advantage.

Extracts the LAST \\boxed{...} in each completion rather than trusting math-verify's default
extraction. Confirmed empirically: when a completion contains more than one \\boxed{...} (e.g.
self-correction -- "first I guess \\boxed{5}, wait, final answer \\boxed{11}"), math-verify's
default any_match extraction returns the set of every boxed value found ({5, 11}), which then
fails verify() against the gold answer even when the model's final answer was correct. Also
confirmed the dataset's raw ground_truth strings must be wrapped in \\boxed{} before parsing --
parsing them unwrapped silently disagrees with the boxed form for expression-type answers
(e.g. "3n", "n^3"), matching only for plain numbers.
"""
from math_verify import parse, verify

# Reward shape lives in config.py: no penalty up to cfg.reward.free_ratio x gold tokens, full penalty at
# cfg.reward.full_ratio x (3x/4x from the 2026-09-18 baseline: p75 of correct answers is 2.8x gold), weighted
# cfg.reward.weight (0.5: a correct answer at >= 4x gold scores 0.5, still above any wrong answer's 0).
from config import cfg


def extract_last_boxed(text: str) -> str | None:
    idx = text.rfind("\\boxed{")
    if idx == -1:
        return None
    start = idx + len("\\boxed{")
    depth, i = 1, start
    while i < len(text) and depth > 0:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1
    return text[start:i - 1] if depth == 0 else None


def _completion_text(completion) -> str:
    if isinstance(completion, str):
        return completion
    return completion[-1]["content"]  # trl conversational format: list of message dicts


def is_correct(completion, gold: str) -> bool:
    boxed = extract_last_boxed(_completion_text(completion))
    if boxed is None:
        return False
    pred = parse(f"\\boxed{{{boxed}}}")
    gold_parsed = parse(f"\\boxed{{{gold}}}")
    return bool(verify(gold_parsed, pred))


def math_reward(completions, ground_truth, **kwargs) -> list[float]:
    """1.0 if the last \\boxed{} answer verifies against ground_truth, else 0.0."""
    return [1.0 if is_correct(c, g) else 0.0 for c, g in zip(completions, ground_truth)]


def length_excess(completion_tokens: int, gold_tokens: int) -> float:
    """0 when completion_tokens <= cfg.reward.free_ratio * gold_tokens, 1 at >= cfg.reward.full_ratio * gold_tokens,
    linear between. gold_tokens <= 0 (missing) disables the penalty for that row."""
    if not gold_tokens or gold_tokens <= 0:
        return 0.0
    ratio = completion_tokens / gold_tokens
    return min(1.0, max(0.0, (ratio - cfg.reward.free_ratio) / (cfg.reward.full_ratio - cfg.reward.free_ratio)))


def length_reward(completions, ground_truth, gold_tokens, completion_ids, **kwargs) -> list[float]:
    """-length_excess for correct completions, 0.0 for wrong ones. Meant to be combined with math_reward at
    weight cfg.reward.weight (trl: reward_funcs=[math_reward, length_reward],
    reward_weights=[1.0, cfg.reward.weight]), giving correct answers 1 - weight*excess and wrong ones 0.
    `gold_tokens` is the dataset column (data/load_and_split.py); `completion_ids` is the per-completion
    token-id list trl passes to every reward function (its length is the completion length in tokens)."""
    rewards = []
    for completion, gold, n_gold, ids in zip(completions, ground_truth, gold_tokens, completion_ids):
        if not is_correct(completion, gold):
            rewards.append(0.0)
            continue
        rewards.append(-length_excess(len(ids), n_gold))
    return rewards


# ---------------------------------------------------------------------------
# The switch is cfg.reward.use_length. Both training scripts build
# their reward list from active_rewards(), so flipping it trains on correctness alone.
# ---------------------------------------------------------------------------


def active_rewards() -> tuple[list, list[float]]:
    """(reward_funcs, reward_weights) for the current switch setting. Feed straight into GRPOConfig
    (reward_weights=) and GRPOTrainer (reward_funcs=), or into total_reward() below."""
    if cfg.reward.use_length:
        return [math_reward, length_reward], [1.0, cfg.reward.weight]
    return [math_reward], [1.0]


def total_reward(completions, ground_truth, **kwargs) -> list[float]:
    """Weighted sum of the active rewards, one float per completion -- what trl computes internally from
    reward_funcs/reward_weights, for the pure-PyTorch script. kwargs must include whatever the active
    functions need (gold_tokens, completion_ids for length_reward); they are ignored by math_reward."""
    funcs, weights = active_rewards()
    per_func = [f(completions, ground_truth, **kwargs) for f in funcs]
    return [sum(w * vals[i] for w, vals in zip(weights, per_func)) for i in range(len(completions))]
