# Determinism of the generation step (review round 1, M3)

**Summary.** Greedy decoding (temperature 0, top-k 1) on this llama.cpp CPU build is **not** byte-reproducible. In every configuration tried, rerunning the same requests changed the text of some traces. In the 30 rerun-vs-committed comparisons below, the scored correctness never changed, but all 30 come from one easy source item, so that is weak evidence. The README's earlier claim "greedy, so runs are reproducible" was wrong and has been removed. Scoring is a different matter: given the committed traces, every number, table and figure regenerates byte-for-byte (`SKIP_GENERATION=1 bash scripts/reproduce.sh`, checked in round 1).

## What was run

All runs used the same llama.cpp `b11815` CPU binary, the same GGUF (sha256 `d01d9952…`), `--swa-full --jinja`, a 512-token reasoning cap and greedy sampling. They all covered the same 6 requests: `mgsm-000` (Janet's ducks) in Swahili and Yoruba × the direct / L2 / En arms. The box was shared and heavily loaded throughout (load average ≈ 21–26).

| run | server | runner | file |
|---|---|---|---|
| committed (Oct 22–23) | `-t 1 -tb 2 -np 2`, two concurrent workers, prompt cache on | `--workers 2` | [`results/traces/tiny-aya-global-q4km.math.jsonl`](../results/traces/tiny-aya-global-q4km.math.jsonl) |
| Peer Reviewer rerun | same as committed | same as committed | [`results/determinism/reviewer_rerun_np2_cache.jsonl`](../results/determinism/reviewer_rerun_np2_cache.jsonl) (+ `.log`) |
| det run 1, det run 2 (back to back, one fresh server) | `-t 1 -tb 2 -np 1`, prompt cache off | `--workers 1` | `results/determinism/det_run{1,2}.jsonl` · script [`scripts/determinism_check.sh`](../scripts/determinism_check.sh) |
| tb1 run 1, tb1 run 2 (back to back, one fresh server) | `-t 1 -tb 1 -np 1`, prompt cache off | `--workers 1` | `results/determinism/det_tb1_run{1,2}.jsonl` · script [`scripts/determinism_check_tb1.sh`](../scripts/determinism_check_tb1.sh) |

The comparisons were made by [`probe/determinism.py`](../probe/determinism.py) and are written to `results/determinism/compare_*.json`. Each comparison reports whether the two texts are byte-identical, the first differing character, the token counts, and correctness under the primary (lenient) parser.

## Results

| comparison | byte-identical | same correctness |
|---|---:|---:|
| committed vs Peer Reviewer rerun (same config) | 2 / 6 | 6 / 6 |
| det run 1 vs det run 2 (`-np 1`, no cache, `-tb 2`) | 1 / 6 | 6 / 6 |
| tb1 run 1 vs tb1 run 2 (`-np 1`, no cache, `-tb 1`) | 3 / 6 | 6 / 6 |
| committed vs det run 1 / det run 2 | 2 / 6, 1 / 6 | 6 / 6, 6 / 6 |
| committed vs tb1 run 1 / tb1 run 2 | 2 / 6, 3 / 6 | 6 / 6, 6 / 6 |
| **all 5 reruns vs committed** | **10 / 30** | **30 / 30** |

What diverges:
- **Direct answers.** `mgsm-000-sw` direct answered `28` in the committed run, det run 1 and both tb1 runs, but `12` in the Peer Reviewer rerun and in det run 2. Both answers are wrong (gold 18), so the score is unchanged.
- **Reasoning traces.** Traces diverge after 30–201 characters, usually as a wording change. For example, `**9**` becomes `9`, or "pancakes" becomes "ackee bread". After that the trace continues on a different path, with up to ±44 tokens of length difference. All diverging reasoning traces still reached the gold 18.

With one slot, no prompt cache and full prompt re-evaluation (the server log shows each prompt fully re-evaluated, e.g. 563 of 563 tokens), outputs still differed between back-to-back runs. Server-side batching and prefix reuse therefore cannot be the only cause. I did not isolate the remaining source. Candidates include run-to-run floating-point differences in the CPU kernels on this heavily loaded box. Chasing it further was not worth the compute on a shared machine.

## What this means for the results

- The committed 660 traces are one sample from a slightly noisy generator, not a canonical output. A rerun will not reproduce them byte-for-byte, and individual borderline items can flip. The 30 / 30 correctness agreement is only for `mgsm-000`, which both reasoning arms solve easily. It says little about harder items, where a wording change early in the trace could plausibly flip the answer.
- This adds generation noise on top of the item-sampling noise that the CIs already capture. It is one more reason to read the small differences in this study (for example, 4 vs 13 discordant low-resource math items) as exploratory.
- The 660 traces were **not** regenerated. A full rerun takes about 4.5 h on the busy box and would still not be canonical.
- For future runs, [`scripts/reproduce.sh`](../scripts/reproduce.sh) and [`probe/run.py`](../probe/run.py) default to the most conservative settings: `-np 1`, `--workers 1` and `cache_prompt: false`. `--cache-prompt` re-enables prefix reuse for speed.
