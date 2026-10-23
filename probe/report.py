"""Render markdown tables from results/summary.json and results/paired.json into
docs/results.md, and splice the headline table into README.md between
<!-- RESULTS:start --> and <!-- RESULTS:end --> so no number is typed by hand."""
from __future__ import annotations

import argparse
import json
import os
import re

from probe.figures import LANG_ORDER
from probe.langs import LANGS
from probe.stats import sig3


def pct(x):
    return "–" if x is None else f"{100 * x:.0f}%"


def ci(c):
    return f"[{100 * c[0]:.0f}, {100 * c[1]:.0f}]"


def per_lang_table(summ, model, task, arms):
    h = ["| language | arm | n | accuracy [95% CI] | L2 rate [95% CI] | English-trace rate | mean tokens [95% CI] | mean rep4 | cap hits | unparsed |",
         "|---|---|---:|---|---|---|---|---|---:|---:|"]
    for l in LANG_ORDER[task]:
        for arm in arms:
            c = summ.get(f"{model}|{arm}|{task}|{l}")
            if not c:
                continue
            if arm == "direct":
                h.append(f"| {LANGS[l]['name']} | {arm} | {c['n']} | {pct(c['acc'])} {ci(c['acc_ci'])} | – | – | – | – | – | {round(c['n'] * (1 - c['parse_rate']))} |")
                continue
            t = c["tokens"]
            r = c["rep4"]
            h.append(f"| {LANGS[l]['name']} | {arm} | {c['n']} | {pct(c['acc'])} {ci(c['acc_ci'])} | {pct(c['l2'])} {ci(c['l2_ci'])} | "
                     f"{pct(c['en_trace'])} | {t[0]:.0f} [{t[1]:.0f}, {t[2]:.0f}] | {r[0]:.3f} | {round(c['cap_hit_rate'] * c['n'])} | "
                     f"{round(c['n'] * (1 - c['parse_rate']))} |")
    return "\n".join(h)


def structure_table(summ, model, task, arms):
    h = ["| language | arm | mean '=' steps [95% CI] | mean list lines [95% CI] |", "|---|---|---|---|"]
    for l in LANG_ORDER[task]:
        for arm in arms:
            c = summ.get(f"{model}|{arm}|{task}|{l}")
            if not c or "eq_steps" not in c:
                continue
            e, li = c["eq_steps"], c["list_lines"]
            h.append(f"| {LANGS[l]['name']} | {arm} | {e[0]:.1f} [{e[1]:.1f}, {e[2]:.1f}] | {li[0]:.1f} [{li[1]:.1f}, {li[2]:.1f}] |")
    return "\n".join(h)


ROW_ORDER = [(t, l) for t in ["math", "piqa"] for l in LANG_ORDER[t]] + \
    [(t, f"group:{g}") for t in ["math", "piqa", "all"] for g in ["control", "high", "low"]]


def ordered(pair, model, a, b):
    for t, sl in ROW_ORDER:
        k = f"{model}|{a}-{b}|{t}|{sl}"
        if k in pair:
            yield k, t, sl, pair[k]


def paired_table(pair, model, a, b):
    adj = a == "l2" and b == "en"
    h = [f"| task | slice | n pairs (clusters) | acc {a} | acc {b} | Δ ({a}−{b}) [95% clustered CI] | {a}-only right | {b}-only right | "
         "cluster sign-flip p | " + ("Holm p | BH q | " if adj else "") + "exact McNemar p (ignores clustering) | Δ tokens [95% CI] |",
         "|---|---|---:|---:|---:|---|---:|---:|---:|" + ("---:|---:|" if adj else "") + "---:|---|"]
    for k, task, sl, v in ordered(pair, model, a, b):
        td = "" if b == "direct" else f"{v['tokens_diff']:+.0f} [{v['tokens_diff_ci'][0]:+.0f}, {v['tokens_diff_ci'][1]:+.0f}]"
        bold = "**" if sl.startswith("group:") else ""
        dup = f" (= {v['duplicate_of'].split('|')[-1].replace('group:', 'pooled ')})" if "duplicate_of" in v else ""
        extra = f"{sig3(v['p_cluster_holm'])} | {sig3(v['p_cluster_bh'])} | " if adj else ""
        h.append(f"| {task} | {bold}{sl.replace('group:', 'pooled ')}{bold}{dup} | {v['n']} ({v['n_clusters']}) | {pct(v['acc_a'])} | {pct(v['acc_b'])} | "
                 f"{100 * v['diff']:+.0f} pp [{100 * v['diff_ci'][0]:+.0f}, {100 * v['diff_ci'][1]:+.0f}] | {v['a_only']} | {v['b_only']} | "
                 f"{sig3(v['p_cluster'])} | {extra}{sig3(v['mcnemar_p'])} | {td} |")
    fam = pair.get(f"_family_{a}-{b}")
    if fam:
        h.append(f"\nHolm / BH adjust the cluster sign-flip p over the family of **{fam['n_tests']} distinct** {a}-vs-{b} tests "
                 "(rows marked “= …” test the same item set as another row and are not counted twice). All comparisons are exploratory; none was pre-registered.")
    return "\n".join(h)


HETERO = {("piqa", "group:low"), ("all", "group:low")}


def glance(summ, pair, model, n_piqa_done, interim):
    def P(task, sl):
        return pair.get(f"{model}|l2-en|{task}|{sl}")

    def fmt(task, sl, v):
        het = ""
        if (task, sl) in HETERO:
            per = [pair.get(f"{model}|l2-en|{t}|{l}") for t in (["math", "piqa"] if task == "all" else [task])
                   for l in ["sw", "yo", "zu"]]
            ds = [p["diff"] for p in per if p]
            if ds and min(ds) < 0 < max(ds):
                het = " · **heterogeneous** (per-language Δ from " + f"{100 * min(ds):+.0f} to {100 * max(ds):+.0f} pp)"
        return (f"{100 * v['diff']:+.0f} pp [{100 * v['diff_ci'][0]:+.0f}, {100 * v['diff_ci'][1]:+.0f}]{het} | "
                f"{v['n']} ({v['n_clusters']}) | {sig3(v['p_cluster'])} | {sig3(v['p_cluster_holm'])} | {sig3(v['mcnemar_p'])} |")
    lines = []
    if interim:
        lines.append(f"> **INTERIM:** Global PIQA generation is still running ({n_piqa_done}/360 traces).\n")
    fam = pair.get("_family_l2-en", {"n_tests": "?"})
    tests = [pair[k] for k in fam.get("tests", [])]
    n_raw = sum(t["p_cluster"] < 0.05 for t in tests)
    n_holm = sum(t["p_cluster_holm"] < 0.05 for t in tests)
    verdict = ("**none is significant** at 0.05 under the cluster-aware test, before or after Holm correction"
               if n_raw == 0 and n_holm == 0 else
               f"{n_raw} reach p<0.05 under the cluster-aware test and {n_holm} after Holm correction")
    lines.append(f"**Exploratory.** No comparison was pre-specified. Across the {fam['n_tests']} distinct L2-vs-En tests, "
                 f"{verdict}. The rows show direction and size, not established effects.\n")
    lines.append("| comparison: L2-forced minus En-forced accuracy | Δ [95% clustered-bootstrap CI] | n pairs (source-item clusters) | cluster sign-flip p (primary) | Holm-adjusted p | exact McNemar p (ignores clustering) |")
    lines.append("|---|---|---|---|---|---|")
    for label, task, sl in [("Math, low-resource tier (sw, yo, zu)", "math", "group:low"),
                            ("Math, French", "math", "fr"),
                            ("Math, English control", "math", "en"),
                            ("Commonsense, Canadian French", "piqa", "fr-CA"),
                            ("Commonsense, French (both locales)", "piqa", "group:high"),
                            ("Commonsense, low-resource tier", "piqa", "group:low"),
                            ("Both tasks, low-resource tier", "all", "group:low"),
                            ("Both tasks, French", "all", "group:high")]:
        v = P(task, sl)
        if v:
            lines.append(f"| {label} | {fmt(task, sl, v)}")
    l2 = summ.get(f"{model}|l2|all|group:low")
    en = summ.get(f"{model}|en|all|group:low")
    if l2 and en:
        lines.append("")
        lines.append(f"L2 rate on low-resource prompts: **{100 * l2['l2']:.0f}%** of L2-forced traces "
                     f"[{100 * l2['l2_ci'][0]:.0f}, {100 * l2['l2_ci'][1]:.0f}] vs {100 * en['l2']:.0f}% of En-forced traces "
                     f"(n={l2['n']} / {en['n']}). Mean generated tokens: {l2['tokens'][0]:.0f} (L2) vs {en['tokens'][0]:.0f} (En). "
                     f"Cap hits: {round(l2['cap_hit_rate'] * l2['n'])} (L2) vs {round(en['cap_hit_rate'] * en['n'])} (En).")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--model", default="tiny-aya-global-q4km")
    ap.add_argument("--arms", default="direct,l2,en")
    ap.add_argument("--readme", default="README.md")
    ap.add_argument("--out", default="docs/results.md")
    ap.add_argument("--piqa-done", type=int, default=None, help="PIQA traces generated (default: count lines)")
    a = ap.parse_args()
    if a.piqa_done is None:
        p = os.path.join(a.results, "traces", f"{a.model}.piqa.jsonl")
        a.piqa_done = sum(1 for _ in open(p, encoding="utf-8")) if os.path.exists(p) else 0
    summ = json.load(open(os.path.join(a.results, "summary.json")))
    pair = json.load(open(os.path.join(a.results, "paired.json")))
    arms = a.arms.split(",")
    src = "Source: [`results/summary.json`](../results/summary.json), [`results/paired.json`](../results/paired.json), generated by [`probe/report.py`](../probe/report.py)."
    doc = [f"# Results tables: {a.model}\n", src + "\n"]
    for task in ["math", "piqa"]:
        doc += [f"## Per language: {task}\n", per_lang_table(summ, a.model, task, arms), ""]
    doc += ["## Trace structure (error analysis): math\n", structure_table(summ, a.model, "math", [x for x in arms if x != "direct"]), ""]
    for x, y in [("l2", "en"), ("l2", "direct"), ("en", "direct")]:
        if x in arms and y in arms:
            doc += [f"## Paired: {x} vs {y}\n", paired_table(pair, a.model, x, y), ""]
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    open(a.out, "w", encoding="utf-8").write("\n".join(doc))
    if os.path.exists(a.readme):
        rd = open(a.readme, encoding="utf-8").read()
        block = ("<!-- RESULTS:start (generated by probe/report.py, do not edit) -->\n"
                 "**Math (MGSM / AfriMGSM)**\n\n" + per_lang_table(summ, a.model, "math", arms) +
                 "\n\n**Commonsense (Global PIQA, 4-way, chance = 25%)**\n\n" + per_lang_table(summ, a.model, "piqa", arms) +
                 "\n\n**Paired L2-forced vs En-forced**\n\n" + paired_table(pair, a.model, "l2", "en") +
                 "\n<!-- RESULTS:end -->")
        rd2 = re.sub(r"<!-- RESULTS:start.*?<!-- RESULTS:end -->", lambda _: block, rd, flags=re.S)
        g = ("<!-- GLANCE:start (generated by probe/report.py, do not edit) -->\n"
             + glance(summ, pair, a.model, a.piqa_done, a.piqa_done < 360) + "\n<!-- GLANCE:end -->")
        rd2 = re.sub(r"<!-- GLANCE:start.*?<!-- GLANCE:end -->", lambda _: g, rd2, flags=re.S)
        open(a.readme, "w", encoding="utf-8").write(rd2)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
