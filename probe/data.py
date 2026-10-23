"""Build the subsampled, parallel evaluation slice.

Items are chosen ONCE by seeded sampling of item indices and the same indices
are used in every language, so every comparison across languages and arms is
paired (same underlying question).

Sources (pinned revisions):
- MGSM (juletxara/mgsm, CC-BY-SA-4.0): en, fr, sw test splits.
- AfriMGSM (masakhane/afrimgsm, Apache-2.0): yor, zul test splits. AfriMGSM is a
  translation of the same 250 MGSM questions (answer_number aligned 250/250,
  checked in `check_alignment`).
- Global PIQA parallel (mrlbenchmarks/global-piqa-parallel, CC-BY-SA-4.0):
  eng_latn, fra_latn_cana (Canadian French), fra_latn_fran, swh_latn, yor_latn,
  zul_latn. 4-way multiple choice; option order is shuffled per language by the
  dataset authors.

Usage: python -m probe.data --raw RAW_DIR --out data/slice.jsonl --n-math 20 --n-piqa 20
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import urllib.request

import pandas as pd

REV = {
    "juletxara/mgsm": "b2f13d426afe3be8d69a7e739b36724db8b66bbc",
    "masakhane/afrimgsm": "8e4268d2b94941f18f63f694cf48e4ae26fbec65",
    "mrlbenchmarks/global-piqa-parallel": "b0b18516a8bc2cb1106bce3dd4db32848ca715ea",
}
MATH_FILES = {
    "en": ("juletxara/mgsm", "en/test-00000-of-00001.parquet"),
    "fr": ("juletxara/mgsm", "fr/test-00000-of-00001.parquet"),
    "sw": ("juletxara/mgsm", "sw/test-00000-of-00001.parquet"),
    "yo": ("masakhane/afrimgsm", "data/yor/test.tsv"),
    "zu": ("masakhane/afrimgsm", "data/zul/test.tsv"),
}
PIQA_FILES = {
    "en": "data/parallel_eng_latn.tsv",
    "fr-CA": "data/parallel_fra_latn_cana.tsv",
    "fr": "data/parallel_fra_latn_fran.tsv",
    "sw": "data/parallel_swh_latn.tsv",
    "yo": "data/parallel_yor_latn.tsv",
    "zu": "data/parallel_zul_latn.tsv",
}
LETTERS = "ABCD"


def fetch(repo: str, path: str, raw: str) -> str:
    local = os.path.join(raw, repo.replace("/", "__"), path)
    if not os.path.exists(local):
        os.makedirs(os.path.dirname(local), exist_ok=True)
        url = f"https://huggingface.co/datasets/{repo}/resolve/{REV[repo]}/{path}"
        urllib.request.urlretrieve(url, local)
    return local


def read(local: str) -> pd.DataFrame:
    return pd.read_parquet(local) if local.endswith(".parquet") else pd.read_csv(local, sep="\t")


def check_alignment(frames: dict[str, pd.DataFrame]) -> None:
    ref = frames["en"].answer_number.astype(float).values
    for k, d in frames.items():
        ok = (d.answer_number.astype(float).values == ref).mean()
        assert ok == 1.0, f"math answers misaligned for {k}: {ok:.3f}"


def build(raw: str, n_math: int, n_piqa: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    math = {k: read(fetch(r, p, raw)) for k, (r, p) in MATH_FILES.items()}
    check_alignment(math)
    math_idx = sorted(rng.sample(range(len(math["en"])), n_math))
    piqa = {k: read(fetch("mrlbenchmarks/global-piqa-parallel", p, raw)) for k, p in PIQA_FILES.items()}
    ex = lambda d: d.example_id.str.extract(r"ex(\d+)")[0]
    common = sorted(set.intersection(*[set(ex(d)) for d in piqa.values()]))
    piqa_ids = sorted(rng.sample(common, n_piqa))
    items = []
    for lang, d in math.items():
        for i in math_idx:
            q = str(d.iloc[i].question).strip()
            items.append(dict(id=f"mgsm-{i:03d}-{lang}", task="math", lang=lang, src_index=i,
                              question=q, gold=str(int(float(d.iloc[i].answer_number)))))
    for lang, d in piqa.items():
        d = d.assign(_ex=ex(d)).set_index("_ex")
        for e in piqa_ids:
            r = d.loc[e]
            opts = [str(r[f"solution{j}"]).strip() for j in range(4)]
            q = str(r.prompt).strip() + "\n" + "\n".join(f"{LETTERS[j]}. {o}" for j, o in enumerate(opts))
            items.append(dict(id=f"piqa-{e}-{lang}", task="piqa", lang=lang, src_index=int(e),
                              question=q, gold=LETTERS[int(r.label)], category=str(r.categories)))
    return items


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="raw")
    ap.add_argument("--out", default="data/slice.jsonl")
    ap.add_argument("--n-math", type=int, default=20)
    ap.add_argument("--n-piqa", type=int, default=20)
    ap.add_argument("--seed", type=int, default=20261022)
    a = ap.parse_args()
    items = build(a.raw, a.n_math, a.n_piqa, a.seed)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    h = hashlib.sha256(open(a.out, "rb").read()).hexdigest()
    print(f"wrote {len(items)} items to {a.out} sha256={h}")


if __name__ == "__main__":
    main()
