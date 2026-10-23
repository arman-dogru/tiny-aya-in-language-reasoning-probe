"""Compare two sets of traces for the same (id, arm) keys: byte-identical completions,
first divergent character, token counts and correctness (lenient parser).
Usage: python -m probe.determinism --a results/traces/X.math.jsonl --b results/determinism/run1.jsonl \
          --label "committed vs reviewer rerun" --out results/determinism/compare_X.json
"""
from __future__ import annotations

import argparse
import json

from probe.metrics import parse_answer_lenient


def load(p):
    return {(r["id"], r["arm"]): r for r in map(json.loads, open(p, encoding="utf-8"))}


def correct(r):
    body = r["completion"] if r["arm"] != "direct" else "#### " + r["completion"]
    return parse_answer_lenient(body, r["task"]) == r["gold"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    x = ap.parse_args()
    A, B = load(x.a), load(x.b)
    rows = []
    for k in sorted(set(A) & set(B)):
        ca, cb = A[k]["completion"], B[k]["completion"]
        div = next((i for i, (u, v) in enumerate(zip(ca, cb)) if u != v), None)
        if div is None and len(ca) != len(cb):
            div = min(len(ca), len(cb))
        rows.append(dict(id=k[0], arm=k[1], identical=ca == cb, first_diff_char=div,
                         tokens_a=A[k]["gen_tokens"], tokens_b=B[k]["gen_tokens"],
                         correct_a=correct(A[k]), correct_b=correct(B[k])))
    out = dict(label=x.label, a=x.a, b=x.b, n=len(rows), identical=sum(r["identical"] for r in rows),
               same_correctness=sum(r["correct_a"] == r["correct_b"] for r in rows), rows=rows)
    json.dump(out, open(x.out, "w"), indent=1)
    print(f"{x.label}: {out['identical']}/{out['n']} byte-identical, {out['same_correctness']}/{out['n']} same correctness")


if __name__ == "__main__":
    main()
