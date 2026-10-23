"""Summarize the hand coding of every discordant low-resource math pair (results/error_coding.csv)
and the per-language structure metrics with bootstrap CIs (M2, m7). Writes results/error_coding_summary.json.

Coding protocol (one coder, the author, after seeing which arm was right; not blinded):
- loser_cause = the first error in the losing trace that, by itself, makes the final answer wrong:
  condition_dropped (a stated quantity or constraint is never used), wrong_operation (right
  quantities, wrong arithmetic operation or sign), wrong_setup (a different problem is solved,
  e.g. summing two options instead of choosing), doomloop_cap (repetition until the token cap),
  arithmetic_error, misread_quantity.
- winner_reasoning_flawed = 1 if the winning trace reaches gold through a visibly wrong step.
"""
import argparse
import csv
import json
from collections import Counter

from probe.stats import bootstrap_mean


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coding", default="results/error_coding.csv")
    ap.add_argument("--per-trace", default="results/per_trace.csv")
    ap.add_argument("--paired", default="results/paired.json")
    ap.add_argument("--model", default="tiny-aya-global-q4km")
    ap.add_argument("--out", default="results/error_coding_summary.json")
    a = ap.parse_args()
    code = list(csv.DictReader(open(a.coding, encoding="utf-8")))
    pair = json.load(open(a.paired))[f"{a.model}|l2-en|math|group:low"]
    want = {("en_only", i) for i in pair["b_only_ids"]} | {("l2_only", i) for i in pair["a_only_ids"]}
    have = {(r["discordant_side"], r["id"]) for r in code}
    assert want == have, f"coding does not match paired.json: missing {want - have}, extra {have - want}"
    out = {"n_coded": len(code)}
    for side in ["en_only", "l2_only"]:
        rs = [r for r in code if r["discordant_side"] == side]
        out[side] = {"n": len(rs), "loser_cause": dict(sorted(Counter(r["loser_cause"] for r in rs).items())),
                     "winner_reasoning_flawed": sum(int(r["winner_reasoning_flawed"]) for r in rs)}
    rows = [r for r in csv.DictReader(open(a.per_trace, encoding="utf-8"))
            if r["model"] == a.model and r["task"] == "math" and r["arm"] in ("l2", "en")]
    struct = {}
    for lang in ["en", "fr", "sw", "yo", "zu"]:
        for arm in ["l2", "en"]:
            rs = [r for r in rows if r["lang"] == lang and r["arm"] == arm]
            for m in ["eq_steps", "list_lines"]:
                v = [float(r[m]) for r in rs]
                mean, lo, hi = bootstrap_mean(v, iters=4000, seed=20261022)
                struct[f"{lang}|{arm}|{m}"] = {"n": len(v), "mean": round(mean, 3), "ci": [round(lo, 3), round(hi, 3)]}
    out["structure_math"] = struct
    json.dump(out, open(a.out, "w"), indent=1, sort_keys=True)
    print(json.dumps({k: v for k, v in out.items() if k != "structure_math"}, indent=1))
    for k, v in struct.items():
        print(k, v)


if __name__ == "__main__":
    main()
