#!/usr/bin/env bash
# End-to-end reproduction of the committed results on CPU.
#   SKIP_GENERATION=1 bash scripts/reproduce.sh
#     rescore the committed traces only: downloads GlotLID (~1.7 GB) + Python deps, keeps the
#     committed data/slice.jsonl, no llama.cpp / GGUF. 74 s after the download on the busy box; output is byte-identical to the committed files.
#   bash scripts/reproduce.sh
#     full run: also downloads llama.cpp b11815 and the 2.1 GB GGUF, rebuilds the slice and
#     regenerates every trace (~4.5 h wall on a contended 8-core box).
# Scoring, stats and figures are deterministic given the traces. Generation is NOT
# byte-reproducible on this llama.cpp CPU build even at temperature 0 (docs/determinism.md);
# the settings below (-np 1, --workers 1, prompt cache off) are the most conservative ones.
set -euo pipefail
T=${TOOLS:-tools}; M=tiny-aya-global-q4km; TR=results/traces
bash scripts/setup.sh
if [ "${SKIP_GENERATION:-0}" != "1" ]; then
  python -m probe.data --raw "$T/raw" --out data/slice.jsonl --n-math 20 --n-piqa 20
fi
python -m probe.lid_check --lid "$T/glotlid.bin" --out results/lid_validation.json
if [ "${SKIP_GENERATION:-0}" != "1" ]; then
  # NOTE: the committed traces were generated with -np 2, --workers 2 and prompt caching on.
  # Future runs use one slot per server, one worker and no prompt cache (see docs/determinism.md).
  for p in 8091 8092; do
    "$T/llama-b11815/llama-server" -m "$T/models/tiny-aya-global-q4_k_m.gguf" -t 1 -tb 1 -c 4096 \
      --port $p -np 1 --swa-full --jinja > "$T/server-$p.log" 2>&1 &
  done
  trap 'kill %1 %2 2>/dev/null || true' EXIT
  for p in 8091 8092; do
    for _ in $(seq 300); do curl -sf "http://127.0.0.1:$p/health" >/dev/null && break; sleep 2; done
    curl -sf "http://127.0.0.1:$p/health" >/dev/null || { echo "server on $p did not become healthy" >&2; exit 1; }
  done
  python -m probe.run --server http://127.0.0.1:8091 --model $M --arms direct,l2,en --tasks math \
    --max-think 512 --workers 1 --out $TR/$M.math.jsonl &
  python -m probe.run --server http://127.0.0.1:8092 --model $M --arms direct,l2,en --tasks piqa \
    --max-think 512 --workers 1 --out $TR/$M.piqa.jsonl &
  wait %3 %4
fi
python -m probe.score --lid "$T/glotlid.bin" --run $TR/$M.math.jsonl --run $TR/$M.piqa.jsonl --out results
python -m probe.trace_lid_check --lid "$T/glotlid.bin" --run $TR/$M.math.jsonl --run $TR/$M.piqa.jsonl
python -m probe.examples --traces $TR/$M.math.jsonl $TR/$M.piqa.jsonl
python -m probe.error_coding
python -m probe.figures --lid "$T/glotlid.bin" --results results --out docs/figures
python -m probe.report
