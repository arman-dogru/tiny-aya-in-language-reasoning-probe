"""Score traces: accuracy, L2 reasoning rate, reasoning tokens, repetition.

Usage: python -m probe.score --lid PATH/glotlid.bin \
          --run results/traces/tiny-aya-global-q4km.jsonl[:think] ... --out results/
Writes results/per_trace.csv, results/summary.json (per cell and pooled, with
95% CIs) and results/paired.json (L2-forced vs En-forced, paired).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict

from probe.langs import LANGS, LENIENT_EXTRA
from probe.metrics import LID, parse_answer, parse_answer_lenient, repetition_score, split_trace, structure
from probe.stats import bootstrap_mean, mcnemar_exact, paired_bootstrap_diff, wilson

GROUPS = {"high": ["fr", "fr-CA"], "low": ["sw", "yo", "zu"], "control": ["en"]}


def score_run(path: str, think: str, lid: LID) -> list[dict]:
    """think: '' (no reasoning tags), 'think' (</think>) or 'aya' (<|END_THINKING|>)."""
    close = {"think": "</think>", "aya": "<|END_THINKING|>"}.get(think, "")
    rows = []
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        # The prefilled opener is NOT model output: exclude it from LID/repetition.
        trace, ans = split_trace(r["completion"], think_tags=bool(close), think_close=close or "</think>")
        if r["arm"] == "direct":
            trace, ans = "", "#### " + r["completion"]
        strict = parse_answer(ans, r["task"])
        body = r["completion"] if r["arm"] != "direct" else "#### " + r["completion"]
        if close and close in body:
            body = body.split(close, 1)[1]  # never read an answer out of the thinking block
        pred = parse_answer_lenient(body, r["task"]) if not (close and close not in r["completion"]) else strict
        lab, p = lid.label(trace) if trace else ("und", 0.0)
        shares = lid.segment_shares(trace) if trace else {}
        tgt = LANGS[r["lang"]]["lid"]
        rows.append(dict(
            id=r["id"], model=r["model"], arm=r["arm"], task=r["task"], lang=r["lang"],
            src=r["id"].rsplit("-", 1)[0] if r["lang"] != "fr-CA" else r["id"][: -len("-fr-CA")],
            gold=r["gold"], pred=pred if pred is not None else "", parsed=int(pred is not None),
            correct=int(pred is not None and pred == r["gold"]),
            format_ok=int(strict is not None), correct_strict=int(strict is not None and strict == r["gold"]),
            gen_tokens=r["gen_tokens"], cap_hit=int(r["cap_hit"]), budget_forced=int(r["budget_forced"]),
            trace_lid=lab, trace_lid_p=round(p, 4),
            l2=int(lab in tgt) if r["arm"] != "direct" else "",
            l2_lenient=int(lab in tgt + LENIENT_EXTRA.get(r["lang"], [])) if r["arm"] != "direct" else "",
            en_trace=int(lab == "eng_Latn") if r["arm"] != "direct" else "",
            target_share=round(sum(v for k, v in shares.items() if k in tgt), 4) if shares else "",
            english_share=round(shares.get("eng_Latn", 0.0), 4) if shares else "",
            rep4=(lambda x: "" if x is None else round(x, 4))(repetition_score(trace)) if trace else "",
            trace_words=len(trace.split()), eq_steps=structure(trace)[0], list_lines=structure(trace)[1],
            wall_s=r["wall_s"],
        ))
    return rows


def cell_summary(rs: list[dict]) -> dict:
    n = len(rs)
    k = sum(r["correct"] for r in rs)
    ks = sum(r["correct_strict"] for r in rs)
    out = dict(n=n, acc=k / n if n else float("nan"), acc_ci=wilson(k, n),
               acc_strict=ks / n if n else float("nan"), acc_strict_ci=wilson(ks, n),
               format_rate=sum(r["format_ok"] for r in rs) / n if n else float("nan"),
               parse_rate=sum(r["parsed"] for r in rs) / n if n else float("nan"),
               cap_hit_rate=sum(r["cap_hit"] for r in rs) / n if n else float("nan"))
    if rs and rs[0]["arm"] != "direct":
        for key in ["l2", "l2_lenient", "en_trace"]:
            kk = sum(int(r[key]) for r in rs)
            out[key] = kk / n
            out[key + "_ci"] = wilson(kk, n)
        out["tokens"] = bootstrap_mean([r["gen_tokens"] for r in rs])
        out["eq_steps"] = bootstrap_mean([r["eq_steps"] for r in rs])
        out["list_lines"] = bootstrap_mean([r["list_lines"] for r in rs])
        out["rep4"] = bootstrap_mean([r["rep4"] for r in rs if r["rep4"] != ""])
        out["lid_hist"] = dict(sorted(((lab, sum(1 for r in rs if r["trace_lid"] == lab)) for lab in
                                       {r["trace_lid"] for r in rs}), key=lambda x: -x[1]))
    return out


def summarize(rows: list[dict]) -> dict:
    by = defaultdict(list)
    for r in rows:
        by[(r["model"], r["arm"], r["task"], r["lang"])].append(r)
        for g, ls in GROUPS.items():
            if r["lang"] in ls:
                by[(r["model"], r["arm"], r["task"], "group:" + g)].append(r)
                by[(r["model"], r["arm"], "all", "group:" + g)].append(r)
    return {"|".join(k): cell_summary(v) for k, v in sorted(by.items())}


def paired(rows: list[dict], a: str, b: str) -> dict:
    idx = {(r["model"], r["arm"], r["id"]): r for r in rows}
    out = {}
    models = sorted({r["model"] for r in rows})
    for m in models:
        cells = defaultdict(list)
        for r in rows:
            if r["model"] != m or r["arm"] != a or (m, b, r["id"]) not in idx:
                continue
            o = idx[(m, b, r["id"])]
            cells[(r["task"], r["lang"])].append((r, o))
            for g, ls in GROUPS.items():
                if r["lang"] in ls:
                    cells[(r["task"], "group:" + g)].append((r, o))
                    cells[("all", "group:" + g)].append((r, o))
        for (task, lang), pairs in sorted(cells.items()):
            ca = [p[0]["correct"] for p in pairs]
            cb = [p[1]["correct"] for p in pairs]
            src = [p[0]["src"] for p in pairs]
            bb = sum(1 for x, y in zip(ca, cb) if x and not y)
            cc = sum(1 for x, y in zip(ca, cb) if y and not x)
            d, lo, hi = paired_bootstrap_diff(ca, cb, clusters=src)
            ta = [p[0]["gen_tokens"] for p in pairs]
            tb = [p[1]["gen_tokens"] for p in pairs]
            td, tlo, thi = paired_bootstrap_diff(ta, tb, clusters=src)
            out[f"{m}|{a}-{b}|{task}|{lang}"] = dict(
                n=len(pairs), acc_a=sum(ca) / len(ca), acc_b=sum(cb) / len(cb), diff=d, diff_ci=(lo, hi),
                a_only=bb, b_only=cc, mcnemar_p=mcnemar_exact(bb, cc),
                tokens_diff=td, tokens_diff_ci=(tlo, thi),
                a_only_ids=[p[0]["id"] for p in pairs if p[0]["correct"] and not p[1]["correct"]],
                b_only_ids=[p[0]["id"] for p in pairs if p[1]["correct"] and not p[0]["correct"]])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lid", required=True)
    ap.add_argument("--run", action="append", required=True, help="path[:think|:aya]")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    lid = LID(a.lid)
    rows = []
    for spec in a.run:
        path, _, flag = spec.partition(":")
        rows += score_run(path, flag, lid)
    keys = [(r["model"], r["id"], r["arm"]) for r in rows]
    dup = len(keys) - len(set(keys))
    assert dup == 0, f"{dup} duplicated (model, id, arm) traces: a resumed run wrote a trace twice"
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "per_trace.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    json.dump(summarize(rows), open(os.path.join(a.out, "summary.json"), "w"), indent=1, ensure_ascii=False)
    pr = {}
    arms = {(r["model"], r["arm"]) for r in rows}
    for m in {r["model"] for r in rows}:
        for x, y in [("l2", "en"), ("l2", "direct"), ("en", "direct"), ("l2", "native"), ("native", "direct")]:
            if (m, x) in arms and (m, y) in arms:
                pr.update(paired([r for r in rows if r["model"] == m], x, y))
    json.dump(pr, open(os.path.join(a.out, "paired.json"), "w"), indent=1, ensure_ascii=False)
    print(f"scored {len(rows)} traces -> {a.out}/per_trace.csv, summary.json, paired.json")


if __name__ == "__main__":
    main()
