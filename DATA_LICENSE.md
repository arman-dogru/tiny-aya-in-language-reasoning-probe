# Data and output licences (per path)

This repository mixes code, third-party dataset text and model outputs, and they are under different terms. The MIT [`LICENSE`](LICENSE) covers **code only**.

| Path | What it is | Terms |
|---|---|---|
| `probe/`, `scripts/`, `tests/`, `*.sh`, `*.py` | code written for this study | MIT ([`LICENSE`](LICENSE)) |
| `data/slice.jsonl` (`mgsm-*` en/fr/sw rows) | subset of MGSM, `juletxara/mgsm` @ `b2f13d4` | CC-BY-SA-4.0 (Shi et al., 2022; derived from GSM8K, Cobbe et al., 2021) |
| `data/slice.jsonl` (`mgsm-*` yo/zu rows) | subset of AfriMGSM, `masakhane/afrimgsm` @ `8e4268d` | Apache-2.0 |
| `data/slice.jsonl` (`piqa-*` rows) | subset of Global PIQA *parallel*, `mrlbenchmarks/global-piqa-parallel` @ `b0b1851` | CC-BY-SA-4.0 **plus the Global PIQA usage restriction below** |
| `results/traces/*.jsonl`, `docs/examples.md`, `results/examples.json`, `results/determinism/*.jsonl` | outputs of Tiny Aya Global (`CohereLabs/tiny-aya-global-GGUF`), which quote the dataset prompts | model outputs under the Tiny Aya CC-BY-NC-4.0 licence (the `tiny-aya-global` card says it "requires also adhering to" the [Cohere Labs Acceptable Use Policy](https://docs.cohere.com/docs/c4ai-acceptable-use-policy)); non-commercial research use only. The quoted dataset text keeps its own licence (rows above) |
| `results/*.json`, `results/*.csv`, `docs/results.md`, `docs/figures/*.png` | statistics, the hand coding (`results/error_coding.csv`) and figures computed for this study | CC-BY-4.0, except quoted dataset or model text, which keeps the terms above |

## Global PIQA usage restriction (repeated from its dataset card)

> Global PIQA is released under a CC BY-SA 4.0 license. However, we do **not** allow training of AI systems on Global PIQA, or on synthetic data that uses Global PIQA as a seed.

Every `piqa-*` row in `data/slice.jsonl` and every PIQA trace in `results/traces/tiny-aya-global-q4km.piqa.jsonl` and `docs/examples.md` falls under this restriction. **Use them for evaluation and analysis only. Do not use them to train or fine-tune any model.**

## Provenance note

Global PIQA *parallel* items were machine-translated from English and then corrected by a native speaker (per its dataset card). The fr-CA and fr-FR PIQA items are therefore post-edited MT of English items, not text originally written in Québec or France. MGSM and AfriMGSM items are human translations of GSM8K questions.

The language-ID model used for scoring, GlotLID (`cis-lmu/glotlid` @ `85cd671`), is downloaded at setup time and is not redistributed here. See its model card for its terms.
