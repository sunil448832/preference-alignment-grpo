"""Greedy vs sampling on the same prompts: how often does sampling solve what greedy could not?

For each holdout prompt: one greedy completion and n samples at the training temperature. Reports
  greedy pass rate, sampled pass@1 (mean over samples), pass@k for k = 1..n (unbiased estimator),
  P(some sample correct | greedy wrong)  -- the question "can sampling rescue a greedy failure"
  P(greedy correct | no sample correct)  -- the reverse leak
and writes per-problem rows (greedy_correct, successes, n) to eval/passk_results.jsonl.
This is also the ceiling GRPO is working toward: RL moves pass@1 toward pass@k, it rarely raises pass@k.

Run from the project root (vLLM only; ~200 x 8 x ~2.5K tokens = ~4M tokens, ~2 h on an A10G):
    python -m eval.passk --sample-size 200 --num-samples 8 --max-new-tokens 4096 [--adapter PATH]
"""
import argparse
import json
from math import comb
from pathlib import Path

from config import cfg
from data.dataset import build_dataset, eval_subset
from eval.common import add_engine_args, load_engine, log
from reward_fn import math_reward

OUT_PATH = Path(__file__).parent / "passk_results.jsonl"


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k for n samples with c correct (Chen et al. 2021)."""
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-size", type=int, default=200, help="deterministic random subset of N holdout rows")
    ap.add_argument("--seed", type=int, default=cfg.eval.seed)
    ap.add_argument("--num-samples", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=cfg.batch.temperature)
    ap.add_argument("--adapter", default=None, help="LoRA checkpoint; default is the base model")
    add_engine_args(ap)
    ap.set_defaults(max_new_tokens=cfg.batch.max_completion_length)
    args = ap.parse_args()
    if args.backend != "vllm":
        raise SystemExit("passk needs --backend vllm (n samples per prompt in one call)")

    ds = build_dataset(f"{cfg.data_dir}/eval_holdout.jsonl")
    ds = eval_subset(ds, args.sample_size, args.seed)
    engine, tokenizer = load_engine(args, adapter_path=args.adapter)
    texts = [tokenizer.apply_chat_template(p, tokenize=False, add_generation_prompt=True) for p in ds["prompt"]]
    n = args.num_samples

    log(f"greedy pass over {len(ds)} prompts")
    greedy = engine.generate_texts(texts, args.max_new_tokens)
    greedy_ok = math_reward([t for t, _ in greedy], ds["ground_truth"])
    log(f"sampling {n} x {len(ds)} at T={args.temperature}")
    sampled = engine.sample_texts(texts, args.max_new_tokens, n, args.temperature)
    rows = []
    for i, (samples, g_ok) in enumerate(zip(sampled, greedy_ok)):
        rewards = math_reward([t for t, _ in samples], [ds["ground_truth"][i]] * n)
        rows.append({
            "problem_id": ds["problem_id"][i], "source": ds["source"][i],
            "greedy_correct": bool(g_ok), "greedy_tokens": greedy[i][1],
            "successes": int(sum(r > 0 for r in rewards)), "n": n,
            "sample_tokens_mean": sum(t for _, t in samples) / n,
            "samples_at_cap": sum(t >= args.max_new_tokens for _, t in samples),
        })
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    N = len(rows)
    g_pass = sum(r["greedy_correct"] for r in rows) / N
    s_pass1 = sum(r["successes"] for r in rows) / (N * n)
    print(f"\nn={N} prompts, {n} samples each at T={args.temperature}, cap {args.max_new_tokens}, backend {engine.name}")
    print(f"greedy pass rate           {g_pass:.3f}")
    print(f"sampled pass@1 (mean)      {s_pass1:.3f}")
    for k in sorted({1, 2, 4, 8, 16, n} & set(range(1, n + 1))):
        print(f"pass@{k:<2d}                    {sum(pass_at_k(n, r['successes'], k) for r in rows) / N:.3f}")
    g_fail = [r for r in rows if not r["greedy_correct"]]
    g_ok = [r for r in rows if r["greedy_correct"]]
    rescued = sum(r["successes"] > 0 for r in g_fail)
    print(f"\ngreedy wrong: {len(g_fail)} prompts; sampling found >=1 success on {rescued} "
          f"-> P(rescue | greedy wrong) = {rescued / max(1, len(g_fail)):.3f}")
    print(f"greedy right: {len(g_ok)} prompts; sampling found 0 successes on {sum(r['successes'] == 0 for r in g_ok)} "
          f"-> P(all samples wrong | greedy right) = {sum(r['successes'] == 0 for r in g_ok) / max(1, len(g_ok)):.3f}")
    mixed = sum(0 < r["successes"] < n for r in rows)
    print(f"groups with mixed outcomes (carry a GRPO gradient at G={n}): {mixed}/{N} = {mixed / N:.2f}")
    print(f"per-problem rows -> {OUT_PATH}")


if __name__ == "__main__":
    main()
