"""Add the gold_tokens column to the existing eval_holdout.jsonl / train_pool.jsonl in place, keyed by
problem_id, so the splits (and the baseline results already tied to them) do not change. New splits
produced by load_and_split.py carry the column already; this is for splits made before it existed.

Run from the project root:  python -m data.add_gold_tokens
"""
import json
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer

from data.load_and_split import MODEL_ID, OUT_DIR

FILES = ["eval_holdout.jsonl", "train_pool.jsonl"]


def main() -> None:
    rows_by_file = {f: [json.loads(line) for line in open(OUT_DIR / f)] for f in FILES}
    wanted = {r["problem_id"] for rows in rows_by_file.values() for r in rows}
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    ds = load_dataset("PrimeIntellect/verifiable-math-problems", split="train")
    gold = {
        ex["problem_id"]: len(tokenizer(ex["gold_standard_solution"] or "")["input_ids"])
        for ex in ds if ex["problem_id"] in wanted
    }
    for f, rows in rows_by_file.items():
        missing = 0
        for r in rows:
            r["gold_tokens"] = gold.get(r["problem_id"], 0)
            missing += r["gold_tokens"] == 0
        with open(OUT_DIR / f, "w") as out:
            for r in rows:
                out.write(json.dumps(r) + "\n")
        vals = [r["gold_tokens"] for r in rows if r["gold_tokens"]]
        print(f"{f}: {len(rows)} rows, gold_tokens median={sorted(vals)[len(vals) // 2]} max={max(vals)}, missing={missing}")


if __name__ == "__main__":
    main()
