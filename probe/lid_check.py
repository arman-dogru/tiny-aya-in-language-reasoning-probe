"""Validate the language-ID model on gold dataset text (human-translated MGSM/AfriMGSM; post-edited MT for Global PIQA) before trusting it on traces.

For every language in the slice, run GlotLID on (a) the slice's question texts and
(b) sentence segments of them, and report how often the label lands in the target
set. Also reports fr-CA vs fr-FR: GlotLID has a single fra_Latn label, so the L2
metric cannot tell Canadian from European French (stated in the README).
Usage: python -m probe.lid_check --lid glotlid.bin --out results/lid_validation.json
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict

from probe.langs import LANGS, LENIENT_EXTRA
from probe.metrics import LID, segments
from probe.stats import wilson


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lid", required=True)
    ap.add_argument("--slice", default="data/slice.jsonl")
    ap.add_argument("--out", default="results/lid_validation.json")
    a = ap.parse_args()
    lid = LID(a.lid)
    items = [json.loads(l) for l in open(a.slice, encoding="utf-8")]
    doc = defaultdict(Counter)
    seg = defaultdict(Counter)
    for it in items:
        text = it["question"] if it["task"] == "math" else it["question"].split("\nA. ")[0]
        doc[it["lang"]][lid.label(text)[0]] += 1
        for s in segments(text):
            lab = lid.label(s)[0]
            if lab != "und":
                seg[it["lang"]][lab] += 1
    out = {}
    for lang in LANGS:
        tgt = LANGS[lang]["lid"]
        len_ = tgt + LENIENT_EXTRA.get(lang, [])
        n, k = sum(doc[lang].values()), sum(v for l, v in doc[lang].items() if l in tgt)
        kl = sum(v for l, v in doc[lang].items() if l in len_)
        ns, ks = sum(seg[lang].values()), sum(v for l, v in seg[lang].items() if l in tgt)
        out[lang] = dict(n_docs=n, doc_acc=k / n, doc_acc_ci=wilson(k, n), doc_acc_lenient=kl / n,
                         n_segments=ns, seg_acc=ks / ns if ns else None, seg_acc_ci=wilson(ks, ns),
                         doc_labels=dict(doc[lang].most_common()), seg_labels=dict(seg[lang].most_common(6)))
    json.dump(out, open(a.out, "w"), indent=1)
    for l, v in out.items():
        print(f"{l:6s} doc {v['doc_acc']:.3f} (n={v['n_docs']}) lenient {v['doc_acc_lenient']:.3f} "
              f"seg {v['seg_acc']:.3f} (n={v['n_segments']}) {v['doc_labels']}")


if __name__ == "__main__":
    main()
