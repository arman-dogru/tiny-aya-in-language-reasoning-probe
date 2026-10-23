#!/usr/bin/env bash
# M3 follow-up: same as determinism_check.sh but with ONE thread for prompt processing too
# (-t 1 -tb 1), to test whether the 2-thread batch path explains the run-to-run differences.
set -euo pipefail
T=${TOOLS:-tools}; PORT=${PORT:-8096}; D=results/determinism
"$T/llama-b11815/llama-server" -m "$T/models/tiny-aya-global-q4_k_m.gguf" -t 1 -tb 1 -c 4096 \
  --port $PORT -np 1 --swa-full --jinja > "$T/server-det-tb1.log" 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null || true' EXIT
until curl -sf localhost:$PORT/health >/dev/null; do sleep 2; done
for i in 1 2; do
  python -m probe.run --server http://127.0.0.1:$PORT --model tiny-aya-global-q4km --arms direct,l2,en \
    --tasks math --langs sw,yo --limit-per-cell 1 --max-think 512 --workers 1 --out $D/det_tb1_run$i.jsonl
done
python -m probe.determinism --a $D/det_tb1_run1.jsonl --b $D/det_tb1_run2.jsonl \
  --label "-t 1 -tb 1 -np 1, no prompt cache: run1 vs run2" --out $D/compare_tb1_run1_run2.json
