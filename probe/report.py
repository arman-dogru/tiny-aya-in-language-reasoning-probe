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


def paired_table(pair, model, a, b):
    h = [f"| task | slice | n pairs | acc {a} | acc {b} | Δ ({a}−{b}) [95% CI] | {a}-only right | {b}-only right | McNemar p | Δ tokens [95% CI] |",
         "|---|---|---:|---:|---:|---|---:|---:|---:|---|"]
    for k, v in pair.items():
        m, ab, task, sl = k.split("|")
        if m != model or ab != f"{a}-{b}":
            continue
        td = "" if b == "direct" else f"{v['tokens_diff']:+.0f} [{v['tokens_diff_ci'][0]:+.0f}, {v['tokens_diff_ci'][1]:+.0f}]"
        bold = "**" if sl.startswith("group:") else ""
        h.append(f"| {task} | {bold}{sl.replace('group:', 'pooled ')}{bold} | {v['n']} | {pct(v['acc_a'])} | {pct(v['acc_b'])} | "
                 f"{100 * v['diff']:+.0f} pp [{100 * v['diff_ci'][0]:+.0f}, {100 * v['diff_ci'][1]:+.0f}] | {v['a_only']} | {v['b_only']} | "
                 f"{v['mcnemar_p']:.2f} | {td} |")
    return "\n".join(h)


def glance(summ, pair, model, n_piqa_done, interim):
    def P(task, sl):
        return pair.get(f"{model}|l2-en|{task}|{sl}")

    def fmt(v):
        return (f"{100 * v['diff']:+.0f} pp [{100 * v['diff_ci'][0]:+.0f}, {100 * v['diff_ci'][1]:+.0f}] "
                f"(n={v['n']} pairs, McNemar p={v['mcnemar_p']:.2f})")
    lines = []
    if interim:
        lines.append(f"> **INTERIM:** Global PIQA generation is still running ({n_piqa_done}/360 traces). "
                     "The math numbers are final; the PIQA and pooled numbers will change.\n")
    lines.append("| question | answer (L2-forced minus En-forced accuracy, 95% clustered-bootstrap CI) |")
    lines.append("|---|---|")
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
            lines.append(f"| {label} | {fmt(v)} |")
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
