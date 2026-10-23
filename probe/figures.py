"""Render every README figure from results/ (per_trace.csv, summary.json, paired.json,
lid_validation.json) and the committed traces. Deterministic: PNG metadata is stripped
so reruns on the same inputs are byte-identical.

Usage: python -m probe.figures --results results --lid glotlid.bin --out docs/figures
"""
from __future__ import annotations

import argparse
import csv
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from probe.langs import LANGS  # noqa: E402

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "svg.hashsalt": "aya", "figure.dpi": 110})
ARM_COL = {"direct": "#9e9e9e", "l2": "#d95f02", "en": "#1b9e77", "native": "#7570b3"}
ARM_NAME = {"direct": "direct (no reasoning)", "l2": "L2-forced reasoning", "en": "En-forced reasoning",
            "native": "native thinking"}
LANG_ORDER = {"math": ["en", "fr", "sw", "yo", "zu"], "piqa": ["en", "fr", "fr-CA", "sw", "yo", "zu"]}
TASK_NAME = {"math": "MGSM / AfriMGSM (math)", "piqa": "Global PIQA (4-way commonsense)"}


def nonneg(e):
    """Error-bar half-widths; clip float noise (e.g. Wilson upper bound 0.9999999 at p=1)."""
    return np.clip(np.array(e, dtype=float), 0, None)


def save(fig, out, name):
    p = os.path.join(out, name)
    fig.savefig(p, metadata={"Software": None}, bbox_inches="tight")
    plt.close(fig)
    print("wrote", p)


def load(results):
    rows = list(csv.DictReader(open(os.path.join(results, "per_trace.csv"), encoding="utf-8")))
    summ = json.load(open(os.path.join(results, "summary.json")))
    pair = json.load(open(os.path.join(results, "paired.json")))
    return rows, summ, pair


def fig_accuracy(summ, model, arms, out):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), gridspec_kw={"width_ratios": [5, 6]})
    for ax, task in zip(axes, ["math", "piqa"]):
        langs = LANG_ORDER[task]
        w = 0.8 / len(arms)
        for j, arm in enumerate(arms):
            xs, ys, lo, hi = [], [], [], []
            for i, l in enumerate(langs):
                c = summ.get(f"{model}|{arm}|{task}|{l}")
                if not c:
                    continue
                xs.append(i + (j - (len(arms) - 1) / 2) * w)
                ys.append(c["acc"])
                lo.append(c["acc"] - c["acc_ci"][0])
                hi.append(c["acc_ci"][1] - c["acc"])
            ax.bar(xs, ys, w, color=ARM_COL[arm], label=ARM_NAME[arm], yerr=nonneg([lo, hi]), capsize=2,
                   error_kw={"lw": 0.8})
            for x, y in zip(xs, ys):
                ax.text(x, 0.01, f"{100 * y:.0f}", ha="center", va="bottom", fontsize=6, color="k", rotation=90)
        n = summ.get(f"{model}|{arms[0]}|{task}|en", {}).get("n", "?")
        if task == "piqa":
            ax.axhline(0.25, ls=":", c="k", lw=0.8)
            ax.text(len(langs) - 0.5, 0.26, "chance", ha="right", fontsize=7)
        ax.set_xticks(range(len(langs)), [LANGS[l]["name"].replace(" (", "\n(") for l in langs])
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("accuracy")
        ax.set_title(f"{TASK_NAME[task]}\nn={n} paired items per language")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=len(arms), fontsize=8,
               frameon=False, bbox_to_anchor=(0.5, -0.08))
    fig.suptitle(f"{model}: accuracy by language and reasoning arm (error bars: Wilson 95% CI)", y=1.07)
    save(fig, out, "accuracy_by_language.png")


def fig_lid_heatmap(summ, model, arms, out):
    cols = [(t, l) for t in ["math", "piqa"] for l in LANG_ORDER[t]]
    arms = [a for a in arms if a != "direct"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 1.4 + 0.5 * len(arms)), gridspec_kw={"wspace": 0.35})
    for ax, key, title in zip(axes, ["l2", "en_trace"], ["trace in PROMPT language (L2 rate)",
                                                         "trace in ENGLISH"]):
        M = np.full((len(arms), len(cols)), np.nan)
        for i, a in enumerate(arms):
            for j, (t, l) in enumerate(cols):
                c = summ.get(f"{model}|{a}|{t}|{l}")
                if c:
                    M[i, j] = c[key]
        im = ax.imshow(M, vmin=0, vmax=1, cmap="viridis", aspect="auto")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if not np.isnan(M[i, j]):
                    ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=7,
                            color="w" if M[i, j] < 0.6 else "k")
        ax.set_xticks(range(len(cols)), [f"{t}\n{l}" for t, l in cols], fontsize=7)
        ax.set_yticks(range(len(arms)), [ARM_NAME[a].replace(" reasoning", "") for a in arms])
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.03)
    fig.suptitle(f"{model}: share of reasoning traces by GlotLID language (whole-trace label)", y=1.05)
    save(fig, out, "trace_language_heatmap.png")


def fig_rep_scatter(rows, arms, out):
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for arm in [a for a in arms if a != "direct"]:
        rs = [r for r in rows if r["arm"] == arm and r["rep4"] != ""]
        x = np.array([int(r["gen_tokens"]) for r in rs])
        y = np.array([float(r["rep4"]) for r in rs])
        cap = np.array([r["cap_hit"] == "1" for r in rs])
        ax.scatter(x[~cap], y[~cap], s=12, alpha=0.6, c=ARM_COL[arm], label=f"{ARM_NAME[arm]} (n={len(rs)})")
        if cap.any():
            ax.scatter(x[cap], y[cap], s=40, marker="x", c=ARM_COL[arm], label=f"{ARM_NAME[arm]}: cap hit ({cap.sum()})")
        if len(x) > 3:
            rho = spearman(x, y)
            ax.text(0.02, 0.62 - 0.07 * list(ARM_COL).index(arm), f"{arm}: Spearman ρ(tokens, rep4) = {rho:.2f}",
                    transform=ax.transAxes, ha="left", fontsize=8, color=ARM_COL[arm])
    ax.set_xscale("log")
    ax.set_xlabel("generated reasoning tokens (log scale)")
    ax.set_ylabel("repetition score (share of repeated word 4-grams)")
    ax.set_title("Doomloop check: repetition vs trace length")
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    save(fig, out, "repetition_vs_length.png")


def spearman(x, y):
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    return float(np.corrcoef(rx, ry)[0, 1])


def fig_forest(pair, model, a, b, out, name):
    keys = [k for k in pair if k.startswith(f"{model}|{a}-{b}|")]
    order = []
    for t in ["math", "piqa"]:
        order += [f"{model}|{a}-{b}|{t}|{l}" for l in LANG_ORDER[t]]
        order += [f"{model}|{a}-{b}|{t}|group:{g}" for g in ["high", "low"]]
    order += [f"{model}|{a}-{b}|all|group:{g}" for g in ["high", "low"]]
    order = [k for k in order if k in keys]
    fig, ax = plt.subplots(figsize=(6.5, 0.28 * len(order) + 1))
    for i, k in enumerate(order[::-1]):
        v = pair[k]
        pooled = "group:" in k
        ax.errorbar(v["diff"], i, xerr=nonneg([[v["diff"] - v["diff_ci"][0]], [v["diff_ci"][1] - v["diff"]]]),
                    fmt="D" if pooled else "o", c="k" if pooled else ARM_COL[a], capsize=2, ms=4)
        _, _, t, l = k.split("|")
        lab = f"{t} · {l.replace('group:', 'POOLED ')}  (n={v['n']}, +{v['a_only']}/−{v['b_only']}, p={v['mcnemar_p']:.2f})"
        ax.text(1.02, i, lab, transform=ax.get_yaxis_transform(), va="center", fontsize=7)
    ax.axvline(0, c="grey", lw=0.8)
    ax.set_yticks([])
    ax.set_xlim(-1, 1)
    ax.set_xlabel(f"accuracy({a}) − accuracy({b}); 95% paired bootstrap CI (clustered by source item)")
    ax.set_title(f"{model}: {ARM_NAME[a]} vs {ARM_NAME[b]}")
    save(fig, out, name)


def fig_tokens(summ, model, arms, out):
    arms = [a for a in arms if a != "direct"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.2), gridspec_kw={"width_ratios": [5, 6]})
    for ax, task in zip(axes, ["math", "piqa"]):
        langs = LANG_ORDER[task]
        w = 0.8 / len(arms)
        for j, arm in enumerate(arms):
            xs, ys, err = [], [], [[], []]
            for i, l in enumerate(langs):
                c = summ.get(f"{model}|{arm}|{task}|{l}")
                if not c:
                    continue
                m, lo, hi = c["tokens"]
                xs.append(i + (j - (len(arms) - 1) / 2) * w)
                ys.append(m)
                err[0].append(m - lo)
                err[1].append(hi - m)
            ax.bar(xs, ys, w, yerr=nonneg(err), color=ARM_COL[arm], label=ARM_NAME[arm], capsize=2, error_kw={"lw": 0.8})
        ax.set_xticks(range(len(langs)), [LANGS[l]["name"].replace(" (", "\n(") for l in langs])
        ax.set_ylabel("mean generated tokens")
        ax.set_title(TASK_NAME[task])
    axes[0].legend(fontsize=7, frameon=False)
    fig.suptitle(f"{model}: reasoning length (bootstrap 95% CI); same tokenizer for both arms", y=1.02)
    save(fig, out, "tokens_by_language.png")


def fig_groups(summ, model, arms, out):
    all_arms = arms
    arms = [a for a in arms if a != "direct"]
    groups = ["control", "high", "low"]
    gname = {"control": "English\n(control)", "high": "French\n(fr + fr-CA)", "low": "low-resource\n(sw, yo, zu)"}
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    for ax, (key, title) in zip(axes, [("acc", "accuracy"), ("l2", "L2 rate (trace in prompt language)"),
                                       ("tokens", "mean generated tokens")]):
        use = all_arms if key == "acc" else arms
        w = 0.8 / len(use)
        for j, arm in enumerate(use):
            xs, ys, err = [], [], [[], []]
            for i, g in enumerate(groups):
                c = summ.get(f"{model}|{arm}|all|group:{g}")
                if not c:
                    continue
                if key == "tokens":
                    m, lo, hi = c["tokens"]
                else:
                    m, (lo, hi) = c[key], c[key + "_ci"]
                xs.append(i + (j - (len(use) - 1) / 2) * w)
                ys.append(m)
                err[0].append(m - lo)
                err[1].append(hi - m)
            ax.bar(xs, ys, w, yerr=nonneg(err), color=ARM_COL[arm], label=ARM_NAME[arm], capsize=2, error_kw={"lw": 0.8})
        ax.set_xticks(range(len(groups)), [gname[g] for g in groups])
        ax.set_title(title)
        if key != "tokens":
            ax.set_ylim(0, 1.05)
    axes[0].legend(fontsize=7, frameon=False)
    fig.suptitle(f"{model}: by resource group, both tasks pooled (95% CIs)", y=1.03)
    save(fig, out, "resource_group_summary.png")


def fig_lid_validation(lidv, out):
    langs = list(lidv)
    fig, ax = plt.subplots(figsize=(6, 2.6))
    for k, (key, lab, col) in enumerate([("doc_acc", "whole question", "#4c72b0"), ("seg_acc", "sentence segments", "#dd8452")]):
        ys = [lidv[l][key] for l in langs]
        ci = [lidv[l][key + "_ci"] for l in langs]
        xs = np.arange(len(langs)) + (k - 0.5) * 0.38
        ax.bar(xs, ys, 0.38, color=col, label=lab,
               yerr=nonneg([[y - c[0] for y, c in zip(ys, ci)], [c[1] - y for y, c in zip(ys, ci)]]), capsize=2)
    ax.set_xticks(range(len(langs)), langs)
    ax.set_ylim(0.5, 1.02)
    ax.set_ylabel("GlotLID label = target")
    ax.set_title("Language-ID sanity check on gold human-written questions")
    ax.legend(fontsize=7, frameon=False, loc="lower left")
    save(fig, out, "lid_validation.png")


def fig_strips(examples, lid, out):
    """Each example trace drawn as a strip of sentence segments coloured by detected language."""
    from probe.metrics import lid_clean, segments  # noqa: PLC0415
    palette = {"eng_Latn": "#1b9e77", "fra_Latn": "#7570b3", "swh_Latn": "#e7298a", "yor_Latn": "#e6ab02",
               "zul_Latn": "#66a61e", "und": "#dddddd"}
    fig, ax = plt.subplots(figsize=(10, 0.55 * len(examples) + 0.8))
    seen = set()
    for i, ex in enumerate(examples):
        x = 0
        segs = segments(ex["trace"])
        tot = sum(max(len(lid_clean(s)), 1) for s in segs) or 1
        for s in segs:
            lab = lid.label(s)[0]
            wdt = max(len(lid_clean(s)), 1) / tot
            col = palette.get(lab, "#d95f02" if lab != "und" else "#dddddd")
            key = lab if lab in palette else "other"
            ax.barh(i, wdt, left=x, color=col, edgecolor="w", lw=0.5, label=None if key in seen else key)
            seen.add(key)
            x += wdt
        ax.text(-0.01, i, ex["label"], ha="right", va="center", fontsize=7)
    ax.set_yticks([])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("position in trace (share of characters); segments coloured by GlotLID label")
    h, lab = ax.get_legend_handles_labels()
    ax.legend(h, lab, ncol=7, fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    ax.set_title("Example traces: which language each sentence is written in")
    save(fig, out, "example_trace_strips.png")


def fig_trace_lid(rows, model, out):
    """Sentence-level check of the whole-trace label: distribution of the prompt-language share
    (L2-forced arm) and English share (En-forced arm) per cell, non-English prompts only."""
    cells = [(t, l) for t in ["math", "piqa"] for l in LANG_ORDER[t] if l != "en"]
    fig, ax = plt.subplots(figsize=(10, 3.4))
    for k, (arm, key) in enumerate([("l2", "target_share"), ("en", "english_share")]):
        data = []
        for t, l in cells:
            v = [float(r[key]) for r in rows if r["model"] == model and r["arm"] == arm and r["task"] == t
                 and r["lang"] == l and r[key] != ""]
            data.append(v if v else [np.nan])
        pos = np.arange(len(cells)) + (k - 0.5) * 0.36
        bp = ax.boxplot(data, positions=pos, widths=0.3, patch_artist=True, showfliers=True,
                        flierprops={"markersize": 3})
        for b in bp["boxes"]:
            b.set_facecolor(ARM_COL[arm])
            b.set_alpha(0.7)
        ax.plot([], [], "s", c=ARM_COL[arm], label=f"{ARM_NAME[arm]}: share of sentences in "
                + ("the prompt language" if arm == "l2" else "English"))
    ax.set_xticks(range(len(cells)), [f"{t}\n{l}" for t, l in cells])
    ax.set_ylim(-0.03, 1.03)
    ax.set_ylabel("character-weighted share of\nsentence segments (GlotLID)")
    ax.set_title("Trace-level language check: is the forced language held sentence by sentence?")
    ax.legend(fontsize=7, frameon=False, loc="lower left")
    save(fig, out, "trace_language_purity.png")


def fig_structure(rows, model, out):
    """Error-analysis view: explicit arithmetic steps ('=' count) per math trace, by language and arm,
    split by whether the answer was right."""
    langs = LANG_ORDER["math"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.2), sharey=True)
    for ax, (key, title) in zip(axes, [("eq_steps", "explicit arithmetic steps ('=' signs) per trace"),
                                       ("list_lines", "numbered / bulleted lines per trace")]):
        for j, arm in enumerate(["l2", "en"]):
            for i, l in enumerate(langs):
                for c, mk, dx in [("1", "o", -0.08), ("0", "x", 0.08)]:
                    v = [float(r[key]) for r in rows if r["model"] == model and r["arm"] == arm and r["task"] == "math"
                         and r["lang"] == l and r["correct"] == c]
                    if v:
                        ax.scatter([i + (j - 0.5) * 0.4 + dx], [np.mean(v)], marker=mk, c=ARM_COL[arm], s=10 + 3 * len(v))
        ax.set_xticks(range(len(langs)), langs)
        ax.set_title(title, fontsize=9)
    axes[0].set_ylabel("mean per trace (MGSM / AfriMGSM)")
    for arm in ["l2", "en"]:
        axes[0].scatter([], [], c=ARM_COL[arm], marker="s", label=ARM_NAME[arm])
    axes[0].scatter([], [], c="k", marker="o", label="correct (marker area ∝ n)")
    axes[0].scatter([], [], c="k", marker="x", label="wrong")
    axes[0].legend(fontsize=7, frameon=False)
    fig.suptitle(f"{model}: how decomposed are the traces? (per language: left pair = L2-forced, right pair = En-forced)", y=1.03)
    save(fig, out, "reasoning_structure.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--lid", required=True)
    ap.add_argument("--model", default="tiny-aya-global-q4km")
    ap.add_argument("--arms", default="direct,l2,en")
    ap.add_argument("--out", default="docs/figures")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rows, summ, pair = load(a.results)
    arms = a.arms.split(",")
    mrows = [r for r in rows if r["model"] == a.model]
    fig_accuracy(summ, a.model, arms, a.out)
    fig_lid_heatmap(summ, a.model, arms, a.out)
    fig_rep_scatter(mrows, arms, a.out)
    fig_forest(pair, a.model, "l2", "en", a.out, "paired_l2_vs_en.png")
    if "direct" in arms:
        fig_forest(pair, a.model, "l2", "direct", a.out, "paired_l2_vs_direct.png")
    fig_tokens(summ, a.model, arms, a.out)
    fig_groups(summ, a.model, arms, a.out)
    fig_trace_lid(rows, a.model, a.out)
    fig_structure(rows, a.model, a.out)
    fig_lid_validation(json.load(open(os.path.join(a.results, "lid_validation.json"))), a.out)
    ex_path = os.path.join(a.results, "examples.json")
    if os.path.exists(ex_path):
        from probe.metrics import LID  # noqa: PLC0415
        fig_strips(json.load(open(ex_path, encoding="utf-8")), LID(a.lid), a.out)


if __name__ == "__main__":
    main()
