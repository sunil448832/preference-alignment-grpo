"""Break a results JSONL down by outcome, truncation, length band, source and gold-answer format.
Answers: is the cap or the model the limit, and how much of the miss is label format rather than reasoning.

Run from the project root:
    python -m eval.analyze_results eval/baseline_results.jsonl [--cap 4096] [--gold-lengths]
--cap defaults to the largest completion_tokens seen (a run where nothing hit the cap will then
misreport the longest row as truncated; pass it explicitly when in doubt).
--gold-lengths also loads the source dataset (cached under HF_HOME) to compare against reference-solution length.
"""
import argparse
import json
import re
import statistics as st
from collections import Counter, defaultdict

from reward_fn import extract_last_boxed


def pct(k, n):
    return f"{k / n:.1%}" if n else "n/a"


def gt_category(g: str) -> str:
    g = g.strip()
    if re.fullmatch(r"-?\d+", g): return "integer"
    if re.fullmatch(r"-?\d*\.\d+", g): return "decimal"
    if re.fullmatch(r"-?\\frac\{-?\d+\}\{\d+\}|-?\d+/\d+", g): return "fraction"
    if re.search(r"\\textbf\{\(?[A-E]\)?\}", g): return "choice_marker+value"
    if re.fullmatch(r"\(?[A-E]\)?", g): return "bare_letter"
    if re.search(r"\\begin\{cases\}|\\text\{[A-Za-z ]{12,}\}|\\quad|\\\\|\bor\b|\band\b", g) or "\n" in g or len(g) > 60:
        return "multi-part/sentence"
    if "%" in g: return "percent"
    if "\\text" in g: return "has_\\text"
    if "," in g: return "list/tuple"
    if "=" in g: return "equation"
    if re.search(r"[a-zA-Z]", g): return "expression"
    return "other"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--cap", type=int, default=None, help="max_new_tokens the run used")
    ap.add_argument("--gold-lengths", action="store_true")
    args = ap.parse_args()

    R = [json.loads(line) for line in open(args.path)]
    n = len(R)
    lens = [r["completion_tokens"] for r in R]
    cap = args.cap or max(lens)
    for r in R:
        r["boxed"] = extract_last_boxed(r["completion"])
        r["trunc"] = r["completion_tokens"] >= cap
        r["cat"] = ("correct" if r["reward"] else
                    "truncated_no_box" if r["trunc"] and r["boxed"] is None else
                    "truncated_boxed_wrong" if r["trunc"] else
                    "no_box" if r["boxed"] is None else "boxed_wrong")

    backends = Counter(r.get("backend", "hf-nf4") for r in R)
    print(f"{args.path}: n={n}  backend={dict(backends)}  cap={cap}")
    print(f"pass rate {sum(r['reward'] for r in R) / n:.3f}   length mean={st.mean(lens):.0f} median={st.median(lens):.0f} "
          f"p90={sorted(lens)[int(.9 * n)]}")

    print("\n== outcome ==")
    for k, v in Counter(r["cat"] for r in R).most_common():
        print(f"  {k:24s} {v:4d}  {pct(v, n)}")
    nt = [r for r in R if not r["trunc"]]
    tr = [r for r in R if r["trunc"]]
    print(f"  truncated: {len(tr)} ({pct(len(tr), n)});  pass among finished: {pct(sum(r['reward'] for r in nt), len(nt))} (n={len(nt)})")

    print("\n== pass rate by completion length ==")
    edges = [0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 6000, 8000, 10 ** 9]
    for lo, hi in zip(edges, edges[1:]):
        if lo >= cap: break
        rs = [r for r in R if lo <= r["completion_tokens"] < min(hi, cap)]
        if rs: print(f"  [{lo:5d},{min(hi, cap):5d}): n={len(rs):4d} pass={pct(sum(r['reward'] for r in rs), len(rs))}")
    print(f"  at cap {cap:5d} : n={len(tr):4d} pass={pct(sum(r['reward'] for r in tr), len(tr))}")

    print("\n== per source ==")
    bs = defaultdict(list)
    for r in R: bs[r["source"]].append(r)
    for s, rs in sorted(bs.items(), key=lambda kv: -len(kv[1])):
        print(f"  {s:14s} n={len(rs):4d} pass={pct(sum(r['reward'] for r in rs), len(rs))} "
              f"trunc={pct(sum(r['trunc'] for r in rs), len(rs))} mean_len={st.mean(r['completion_tokens'] for r in rs):.0f}")

    print("\n== gold-answer format (share / pass) ==")
    bc = defaultdict(list)
    for r in R: bc[gt_category(r["ground_truth"])].append(r)
    for c, rs in sorted(bc.items(), key=lambda kv: -len(kv[1])):
        print(f"  {c:22s} n={len(rs):4d} pass={pct(sum(r['reward'] for r in rs), len(rs))}")
    unwinnable = [r for r in R if gt_category(r["ground_truth"]) in ("choice_marker+value", "bare_letter", "multi-part/sentence")]
    print(f"  -> {len(unwinnable)} ({pct(len(unwinnable), n)}) have a gold format the verifier/prompt can't reliably score")

    bw = [r for r in R if r["cat"] == "boxed_wrong"]
    if bw:
        print(f"\n== finished-but-wrong ({len(bw)}), first 15: pred vs gold ==")
        for r in bw[:15]:
            print(f"  [{r['source'][:9]:9s}] pred={r['boxed'][:36]!r:40s} gold={r['ground_truth'][:36]!r}")

    if args.gold_lengths:
        from datasets import load_dataset
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507")
        ids = {r["problem_id"] for r in R}
        ds = load_dataset("PrimeIntellect/verifiable-math-problems", split="train")
        gold = {ex["problem_id"]: ex["gold_standard_solution"] for ex in ds if ex["problem_id"] in ids}
        gl = {pid: len(tok(sol)["input_ids"]) for pid, sol in gold.items() if sol}
        vals = list(gl.values())
        print(f"\n== gold_standard_solution length (n={len(vals)}) ==")
        print(f"  mean={st.mean(vals):.0f} median={st.median(vals):.0f} max={max(vals)}  over cap: {sum(v >= cap for v in vals)}")
        ratio = [r["completion_tokens"] / gl[r["problem_id"]] for r in nt if gl.get(r["problem_id"])]
        if ratio: print(f"  model/gold length ratio, finished rows: median {st.median(ratio):.1f}x")
        tg = [gl[r["problem_id"]] for r in tr if gl.get(r["problem_id"])]
        if tg: print(f"  gold length for truncated rows: mean={st.mean(tg):.0f} max={max(tg)}")


if __name__ == "__main__":
    main()
