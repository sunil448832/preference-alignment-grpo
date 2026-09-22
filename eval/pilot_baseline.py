"""Stage 1 -- pilot: measure baseline headroom before any training.
docs/PLAN.md "Stage 1 -- Pilot: measure baseline headroom".
Run from the project root as a module:
    python -m eval.pilot_baseline [--sample-size N] [--seed S] [--max-new-tokens 2048] [--backend vllm|hf] [--quant bnb|bf16]
--sample-size N scores a deterministic random subset of N holdout rows (data.dataset.eval_subset, seed S,
default cfg.eval.seed): the same rows as the trainer's in-loop eval and as evaluate.py for the same N and S.
(`python eval/pilot_baseline.py` puts eval/ on sys.path instead of the project root, and the
`data` / `eval` imports below fail).
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

from config import cfg
from data.dataset import build_dataset, eval_subset
from eval.common import add_engine_args, generate_and_score, load_engine, log, resolve_batch_size

OUT_PATH = Path(__file__).parent / "baseline_results.jsonl"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-size", type=int, default=None, help="deterministic random subset of N holdout rows (default: all)")
    ap.add_argument("--seed", type=int, default=cfg.eval.seed, help="seed of the subset draw (default: cfg.eval.seed)")
    add_engine_args(ap)
    args = ap.parse_args()

    engine, tokenizer = load_engine(args)
    eval_dataset = build_dataset("data/eval_holdout.jsonl")
    eval_dataset = eval_subset(eval_dataset, args.sample_size, args.seed)
    log(f"loaded {len(eval_dataset)} eval examples")

    results = generate_and_score(engine, tokenizer, eval_dataset, batch_size=resolve_batch_size(args),
                                 max_new_tokens=args.max_new_tokens, desc="baseline")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    log(f"wrote {len(results)} results to {OUT_PATH}")

    overall = sum(r["reward"] for r in results) / len(results)
    by_source = defaultdict(list)
    for r in results:
        by_source[r["source"]].append(r["reward"])

    print(f"overall baseline pass rate: {overall:.3f}  (n={len(results)})")
    for source, rewards in sorted(by_source.items()):
        print(f"  {source:15s}: {sum(rewards) / len(rewards):.3f}  (n={len(rewards)})")

    if overall > 0.8:
        print("\nWARNING: baseline pass rate > 0.8 -- likely saturated, no headroom for GRPO.")
        print("Re-filter data/load_and_split.py's HARD_SOURCES to drop the easiest sources above.")
    elif overall < 0.2:
        print("\nWARNING: baseline pass rate < 0.2 -- likely too hard, GRPO will see mostly zero reward.")
    else:
        print("\nbaseline pass rate is in the useful 0.2-0.8 band.")


if __name__ == "__main__":
    main()
