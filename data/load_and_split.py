"""Stage: data prep.

Pull PrimeIntellect/verifiable-math-problems, keep only the competition-level sources,
and carve a held-out eval split from the training pool.

Sources kept: olympiads, amc_aime, synthetic_amc, aops_forum (harder, more likely to sit
below ceiling for a 4B model). Sources dropped: gsm8k, math, cn_k12, orca_math, synthetic_math
-- gsm8k and math are the exact benchmarks Qwen decontaminates its training corpus against
(docs/PLAN.md's "why not public benchmark datasets"); cn_k12/orca_math/synthetic_math are grade-school
level and likely near-ceiling out of the box.
"""
import ast
import json
import random
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer

HARD_SOURCES = {"olympiads", "amc_aime", "synthetic_amc", "aops_forum"}
EVAL_SIZE = 500
POOL_SIZE = 5000
SEED = 0
MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"  # gold_tokens are counted with the policy's tokenizer

OUT_DIR = Path(__file__).parent


def parse_ground_truth(verification_info: str) -> str | None:
    try:
        info = ast.literal_eval(verification_info)
    except (ValueError, SyntaxError):
        return None
    return info.get("ground_truth") or None


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def main() -> None:
    ds = load_dataset("PrimeIntellect/verifiable-math-problems", split="train")
    ds = ds.filter(lambda ex: ex["source"] in HARD_SOURCES)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    rows = []
    for ex in ds:
        ground_truth = parse_ground_truth(ex["verification_info"])
        if ground_truth is None:
            continue
        rows.append({
            "problem_id": ex["problem_id"],
            "source": ex["source"],
            "prompt": ex["prompt"],
            "ground_truth": ground_truth,
            # reference-solution length in policy tokens; drives reward_fn.length_reward (0 = unknown)
            "gold_tokens": len(tokenizer(ex["gold_standard_solution"] or "")["input_ids"]),
        })

    rng = random.Random(SEED)
    rng.shuffle(rows)

    eval_rows = rows[:EVAL_SIZE]
    pool_rows = rows[EVAL_SIZE:EVAL_SIZE + POOL_SIZE]

    write_jsonl(OUT_DIR / "eval_holdout.jsonl", eval_rows)
    write_jsonl(OUT_DIR / "train_pool.jsonl", pool_rows)

    print(f"kept {len(rows)} rows after source filter + verification_info parse")
    print(f"eval_holdout.jsonl: {len(eval_rows)} rows")
    print(f"train_pool.jsonl:   {len(pool_rows)} rows")


if __name__ == "__main__":
    main()
