"""Stage: dataset prep for trl.GRPOTrainer.

Wraps the JSONL files from load_and_split.py into a datasets.Dataset with the column
GRPOTrainer requires: "prompt", in conversational format ([{"role": "user", "content": ...}])
so the tokenizer's chat template is applied automatically instead of feeding the model raw
text -- Qwen3-4B-Instruct-2507 is a chat model, not a base model, and needs its template's
special tokens for the instruct behavior it was tuned for.

Extra columns (ground_truth, source, problem_id) are left in place and reach reward_fn.math_reward
as kwargs -- confirmed against trl 1.13.0's actual source (GRPOTrainer builds reward_kwargs from
every dataset column except prompt/completion/completion_ids), which contradicts what
GRPOTrainer's own docstring claims ("additional columns ... ignored"). Trusting the code over
the docstring here; verify this still holds if trl is upgraded.
"""
from datasets import Dataset, load_dataset


def build_dataset(jsonl_path: str) -> Dataset:
    ds = load_dataset("json", data_files=jsonl_path, split="train")
    return ds.map(lambda ex: {"prompt": [{"role": "user", "content": ex["prompt"]}]})


def load_grpo_datasets(data_dir: str = "data") -> tuple[Dataset, Dataset]:
    train_ds = build_dataset(f"{data_dir}/train_pool.jsonl")
    eval_ds = build_dataset(f"{data_dir}/eval_holdout.jsonl")
    return train_ds, eval_ds


def eval_subset(ds: Dataset, sample_size: int | None, seed: int = 0) -> Dataset:
    """A deterministic random subset of `sample_size` rows: the same rows for the same (size, seed) on every
    machine and every call, chosen with a private RNG so training's torch RNG (and resume) is untouched.
    None or a size >= len(ds) returns the whole set. Indices are sorted so the file order is kept within the
    subset. Used by the in-loop eval and by the eval scripts, so all of them score the identical prompts."""
    import random
    if sample_size is None or sample_size >= len(ds):
        return ds
    idx = sorted(random.Random(seed).sample(range(len(ds)), sample_size))
    return ds.select(idx)
