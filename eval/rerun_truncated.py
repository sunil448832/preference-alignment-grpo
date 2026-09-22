"""Rerun only the problems whose baseline completion hit the token cap, with a larger cap.
Answers "is the cap the limit, or the model?" without re-evaluating the whole holdout.
Reads eval/baseline_results.jsonl, reruns every row with completion_tokens >= the baseline cap,
and reports how many now finish, at what length, and how many pass.

Run from the project root:  python -m eval.rerun_truncated [--max-new-tokens 4096] [--adapter PATH] [--backend vllm|hf]
"""
import argparse
import json
from pathlib import Path

from datasets import Dataset

from eval.common import add_engine_args, generate_and_score, load_engine, log, resolve_batch_size

BASELINE_PATH = Path(__file__).parent / "baseline_results.jsonl"
OUT_PATH = Path(__file__).parent / "rerun_truncated_results.jsonl"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=None, help="LoRA checkpoint; default is the base model")
    add_engine_args(ap)
    ap.set_defaults(max_new_tokens=4096)
    args = ap.parse_args()

    baseline = [json.loads(line) for line in open(BASELINE_PATH)]
    old_cap = max(r["completion_tokens"] for r in baseline)
    truncated_ids = {r["problem_id"] for r in baseline if r["completion_tokens"] >= old_cap}
    holdout = [json.loads(line) for line in open("data/eval_holdout.jsonl")]
    subset = Dataset.from_list([row for row in holdout if row["problem_id"] in truncated_ids])
    log(f"baseline cap was {old_cap}; {len(subset)} problems hit it; rerunning at {args.max_new_tokens}")

    engine, tokenizer = load_engine(args, adapter_path=args.adapter)
    results = generate_and_score(
        engine, tokenizer, subset, batch_size=resolve_batch_size(args), max_new_tokens=args.max_new_tokens,
        desc="rerun",
    )
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    log(f"wrote {len(results)} results to {OUT_PATH}")

    n = len(results)
    finished = [r for r in results if r["completion_tokens"] < args.max_new_tokens]
    passed = sum(r["reward"] for r in results)
    print(f"\nrerun at {args.max_new_tokens}: n={n}  finished={len(finished)} ({len(finished)/n:.0%})  "
          f"passed={passed:.0f} ({passed/n:.0%})  still truncated={n - len(finished)}")
    print("finished-length distribution (tells you whether a smaller cap would have done):")
    for lo, hi in [(old_cap, 2560), (2560, 3072), (3072, 3584), (3584, args.max_new_tokens)]:
        rs = [r for r in finished if lo <= r["completion_tokens"] < hi]
        if rs:
            print(f"  [{lo},{hi}): {len(rs):3d} finished, {sum(r['reward'] for r in rs):.0f} passed")
    baseline_pass = sum(r["reward"] for r in baseline)
    print(f"\nwhole-holdout pass rate if these results replace the baseline rows: "
          f"{(baseline_pass - sum(r['reward'] for r in baseline if r['problem_id'] in truncated_ids) + passed) / len(baseline):.3f}")


if __name__ == "__main__":
    main()
