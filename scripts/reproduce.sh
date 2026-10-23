#!/usr/bin/env bash
# End-to-end reproduction of the committed results on CPU.
# Generation is the slow part: ~4.5 h wall on a contended 8-core box with two
# single-thread llama.cpp servers (see README "Compute"). Everything after
# generation takes < 1 min and is deterministic.
set -euo pipefail
T=${TOOLS:-tools}; M=tiny-aya-global-q4km; TR=results/traces
bash scripts/setup.sh
python -m probe.data --raw "$T/raw" --out data/slice.jsonl --n-math 20 --n-piqa 20
python -m probe.lid_check --lid "$T/glotlid.bin" --out results/lid_validation.json
if [ "${SKIP_GENERATION:-0}" != "1" ]; then
  for p in 8091 8092; do
    "$T/llama-b11815/llama-server" -m "$T/models/tiny-aya-global-q4_k_m.gguf" -t 1 -tb 2 -c 6144 \
      --port $p -np 2 --swa-full --jinja > "$T/server-$p.log" 2>&1 &
  done
  sleep 30
  python -m probe.run --server http://127.0.0.1:8091 --model $M --arms direct,l2,en --tasks math \
    --max-think 512 --workers 2 --out $TR/$M.math.jsonl &
  python -m probe.run --server http://127.0.0.1:8092 --model $M --arms direct,l2,en --tasks piqa \
    --max-think 512 --workers 2 --out $TR/$M.piqa.jsonl &
  wait %3 %4
  kill %1 %2
fi
python -m probe.score --lid "$T/glotlid.bin" --run $TR/$M.math.jsonl --run $TR/$M.piqa.jsonl --out results
python -m probe.trace_lid_check --lid "$T/glotlid.bin" --run $TR/$M.math.jsonl --run $TR/$M.piqa.jsonl
python -m probe.examples --traces $TR/$M.math.jsonl $TR/$M.piqa.jsonl
python -m probe.figures --lid "$T/glotlid.bin" --results results --out docs/figures
python -m probe.report
