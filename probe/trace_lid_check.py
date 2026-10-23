"""Language-ID checks on the *reasoning traces* (not just gold text).

For each (model, arm, task, lang) cell of reasoning traces it reports:
- mean GlotLID confidence of the whole-trace label and the share with confidence < 0.5,
- agreement between the whole-trace label and the majority sentence-segment label,
- mean prompt-language share and English share over sentence segments (code-switching),
- the share of traces whose L2 decision would flip under the lenient Nguni set (Zulu).
Writes results/trace_lid_check.json. Usage:
  python -m probe.trace_lid_check --lid glotlid.bin --run results/traces/X.jsonl ...
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict

from probe.langs import LANGS
from probe.metrics import LID, segments, split_trace


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lid", required=True)
    ap.add_argument("--run", action="append", required=True)
    ap.add_argument("--out", default="results/trace_lid_check.json")
    a = ap.parse_args()
    lid = LID(a.lid)
    cells = defaultdict(list)
    for p in a.run:
        for line in open(p, encoding="utf-8"):
            r = json.loads(line)
            if r["arm"] == "direct":
                continue
            trace, _ = split_trace(r["completion"], think_tags=False)
            lab, prob = lid.label(trace)
            seg = Counter()
            for s in segments(trace):
                sl = lid.label(s)[0]
                if sl != "und":
                    seg[sl] += 1
            maj = seg.most_common(1)[0][0] if seg else "und"
            cells[(r["model"], r["arm"], r["task"], r["lang"])].append((lab, prob, maj))
    out = {}
    for (m, arm, t, l), v in sorted(cells.items()):
        n = len(v)
        tgt = LANGS[l]["lid"]
        out[f"{m}|{arm}|{t}|{l}"] = dict(
            n=n, mean_conf=sum(x[1] for x in v) / n, low_conf_share=sum(x[1] < 0.5 for x in v) / n,
            whole_vs_segment_majority_agree=sum(x[0] == x[2] for x in v) / n,
            l2_whole=sum(x[0] in tgt for x in v) / n, l2_segment_majority=sum(x[2] in tgt for x in v) / n,
            labels=dict(Counter(x[0] for x in v).most_common()))
    json.dump(out, open(a.out, "w"), indent=1)
    worst = sorted(out.items(), key=lambda kv: kv[1]["whole_vs_segment_majority_agree"])[:5]
    tot = sum(v["n"] for v in out.values())
    agree = sum(v["whole_vs_segment_majority_agree"] * v["n"] for v in out.values()) / tot
    low = sum(v["low_conf_share"] * v["n"] for v in out.values()) / tot
    print(f"{tot} traces: whole-vs-segment agreement {agree:.3f}, low-confidence share {low:.3f}")
    for k, v in worst:
        print(" lowest agreement:", k, round(v["whole_vs_segment_majority_agree"], 3), v["labels"])


if __name__ == "__main__":
    main()
