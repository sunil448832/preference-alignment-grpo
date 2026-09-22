"""Post-mortem of a training run from its two logs: did anything move, and was there signal to move it?

    python -m eval.analyze_run checkpoints/grpo_pytorch [--window 25]

From eval_log.jsonl (per-problem results): the pass-rate series, paired flips first->last (net solved vs lost),
pass@any-checkpoint (how many distinct problems some checkpoint solved: the greedy-eval "knowledge set"), and
the always-solved core. From train_log.jsonl: per-window means of reward, zero-variance groups, rescued/parked,
sequences actually trained on, tokens, at-cap -- i.e. how much gradient each window carried.
"""
import argparse
import json
import sys
from collections import defaultdict


def load(path):
    try:
        return [json.loads(l) for l in open(path) if l.strip()]
    except FileNotFoundError:
        return []


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--window", type=int, default=25)
    a = ap.parse_args()

    ev = load(f"{a.run_dir}/eval_log.jsonl")
    ev = [r for r in ev if "results" in r]
    if ev:
        print(f"== eval_log: {len(ev)} evals on {ev[0]['n']} prompts (seed {ev[0].get('seed')}) ==")
        first, last = ev[0], ev[-1]
        ids = list(first["results"])
        solved_by = {pid: [int(r["results"].get(pid, [0])[0]) for r in ev] for pid in ids}
        n = len(ids)
        print(f"  pass rate first {first['pass_rate']:.3f} (step {first['step']}) -> last {last['pass_rate']:.3f} (step {last['step']})")
        gained = sum(1 for pid in ids if solved_by[pid][0] == 0 and solved_by[pid][-1] == 1)
        lost = sum(1 for pid in ids if solved_by[pid][0] == 1 and solved_by[pid][-1] == 0)
        print(f"  paired first->last: +{gained} solved, -{lost} lost, net {gained - lost:+d} of {n}")
        any_ = sum(1 for pid in ids if any(solved_by[pid])); always = sum(1 for pid in ids if all(solved_by[pid]))
        never = sum(1 for pid in ids if not any(solved_by[pid]))
        print(f"  pass@any checkpoint {any_ / n:.3f} | always solved {always / n:.3f} | never solved {never / n:.3f}")
        print(f"    (per-eval pass rate sits between 'always' and 'any'; the gap is greedy flip noise: a single eval "
              f"can move by ~{(any_ - always) / n / 2:.2f} without the model changing)")
        # first-half vs second-half means, per problem: a small consistent gain shows up here before it shows in one eval
        h = len(ev) // 2
        if h >= 2:
            fh = sum(sum(solved_by[pid][:h]) for pid in ids) / (h * n); sh = sum(sum(solved_by[pid][h:]) for pid in ids) / ((len(ev) - h) * n)
            print(f"  mean pass rate, first {h} evals {fh:.3f} vs last {len(ev) - h} evals {sh:.3f} -> {sh - fh:+.3f}")
        tl = [(r["step"], r["mean_tokens"], r["frac_at_cap"]) for r in ev]
        print(f"  mean tokens {tl[0][1]:.0f} -> {tl[-1][1]:.0f} | at cap {tl[0][2]:.2f} -> {tl[-1][2]:.2f}")
    else:
        print("no eval_log with per-problem results")

    tr = load(f"{a.run_dir}/train_log.jsonl")
    if tr:
        print(f"\n== train_log: {len(tr)} steps, windows of {a.window} ==")
        keys = ["mean_reward", "zero_variance_groups", "active_sequences", "rescued_prompts", "exhausted_prompts",
                "parked_total", "mean_completion_tokens", "frac_at_cap", "step_s"]
        print(f"  {'steps':>9} " + " ".join(f"{k[:11]:>11}" for k in keys))
        for s in range(0, len(tr), a.window):
            w = tr[s:s + a.window]
            vals = []
            for k in keys:
                v = [r.get(k) for r in w if r.get(k) is not None]
                vals.append(sum(v) / len(v) if v else float("nan"))
            print(f"  {w[0]['step']:4d}-{w[-1]['step']:4d} " + " ".join(f"{v:11.3f}" for v in vals))
        total_active = sum(r.get("active_sequences", 0) for r in tr); total_seq = sum(r.get("active_sequences", 0) + 0 for r in tr)
        skipped = sum(1 for r in tr if r.get("skipped_backprop"))
        print(f"  steps with no gradient at all: {skipped}/{len(tr)} | sequences trained on in total: {total_active}")
        print("  (mean_reward is biased by retries and parking: rescued groups add successes, parked prompts leave the draw)")


if __name__ == "__main__":
    main()
