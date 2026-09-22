"""Before/after comparison on the held-out eval split. docs/PLAN.md "Evaluation".
Run after training, from the project root:
    python evaluate.py checkpoints/grpo_rlvr [--sample-size N] [--seed S] [--max-new-tokens 2048] [--backend vllm|hf]
    (adapter path may be checkpoints/grpo_pytorch/step_500 etc.; use the same subset and cap as the baseline)
Requires eval/baseline_results.jsonl to already exist (run `python -m eval.pilot_baseline` first).
"""
import argparse
import json
from pathlib import Path

from config import cfg
from data.dataset import build_dataset, eval_subset
from eval.common import add_engine_args, generate_and_score, load_engine, log, resolve_batch_size

BASELINE_PATH = Path(__file__).parent / "eval" / "baseline_results.jsonl"
AFTER_PATH = Path(__file__).parent / "eval" / "after_results.jsonl"


def load_baseline() -> list[dict]:
    with open(BASELINE_PATH) as f:
        return [json.loads(line) for line in f]


def summarize(name: str, results: list[dict]) -> tuple[float, float]:
    pass_rate = sum(r["reward"] for r in results) / len(results)
    mean_len = sum(r["completion_tokens"] for r in results) / len(results)
    print(f"{name:10s} | pass rate {pass_rate:.3f} | mean completion length {mean_len:.1f} tok  (n={len(results)})")
    return pass_rate, mean_len


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("adapter_path", help="path to trained LoRA adapter checkpoint")
    ap.add_argument("--sample-size", type=int, default=None, help="deterministic random subset of N holdout rows (default: all)")
    ap.add_argument("--seed", type=int, default=cfg.eval.seed, help="seed of the subset draw (default: cfg.eval.seed)")
    add_engine_args(ap)
    args = ap.parse_args()
    adapter_path = args.adapter_path

    before = load_baseline()
    eval_dataset = build_dataset("data/eval_holdout.jsonl")
    eval_dataset = eval_subset(eval_dataset, args.sample_size, args.seed)
    # Pair on problem_id so a baseline run over 500 still compares fairly against a subset run.
    keep = set(eval_dataset["problem_id"])
    before = [r for r in before if r["problem_id"] in keep]
    if len(before) != len(keep):
        raise SystemExit(f"baseline has {len(before)} of the {len(keep)} requested problems; rerun eval.pilot_baseline")
    before_pass_rate, before_len = summarize("before", before)

    baseline_backends = {r.get("backend", "hf-nf4") for r in before}
    engine, tokenizer = load_engine(args, adapter_path=adapter_path)
    if baseline_backends != {engine.name}:
        print(f"WARNING: baseline was run with backend {sorted(baseline_backends)}, this run uses {engine.name}; "
              "the delta below mixes an inference-engine change with the training effect")
    log(f"loaded {len(eval_dataset)} eval examples")
    after = generate_and_score(engine, tokenizer, eval_dataset, batch_size=resolve_batch_size(args),
                               max_new_tokens=args.max_new_tokens, desc="after")
    after_pass_rate, after_len = summarize("after", after)

    print(f"\npass-rate delta: {after_pass_rate - before_pass_rate:+.3f}")
    length_ratio = after_len / before_len
    warn = "  -- WARNING: response length grew >50%, check for length-gaming (docs/PLAN.md, Evaluation)" \
        if length_ratio > 1.5 else ""
    print(f"completion length ratio (after/before): {length_ratio:.2f}x{warn}")

    AFTER_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(AFTER_PATH, "w") as f:
        for r in after:
            f.write(json.dumps(r) + "\n")
    print(f"\nsaved per-example after-results to {AFTER_PATH}")


if __name__ == "__main__":
    main()
