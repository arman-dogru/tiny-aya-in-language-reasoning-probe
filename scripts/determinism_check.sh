#!/usr/bin/env bash
# M3 check: run the same 6-trace slice twice with the deterministic configuration
# (server -np 1, runner --workers 1, cache_prompt false) and compare byte-for-byte.
# Also compares the committed traces against the reviewer's rerun under the original
# (-np 2, --workers 2, cache_prompt true) configuration.
set -euo pipefail
T=${TOOLS:-tools}; PORT=${PORT:-8095}; D=results/determinism
"$T/llama-b11815/llama-server" -m "$T/models/tiny-aya-global-q4_k_m.gguf" -t 1 -tb 2 -c 4096 \
  --port $PORT -np 1 --swa-full --jinja > "$T/server-det.log" 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null || true' EXIT
until curl -sf localhost:$PORT/health >/dev/null; do sleep 2; done
for i in 1 2; do
  python -m probe.run --server http://127.0.0.1:$PORT --model tiny-aya-global-q4km --arms direct,l2,en \
    --tasks math --langs sw,yo --limit-per-cell 1 --max-think 512 --workers 1 --out $D/det_run$i.jsonl
done
python -m probe.determinism --a $D/det_run1.jsonl --b $D/det_run2.jsonl \
  --label "deterministic config: run1 vs run2" --out $D/compare_det_run1_run2.json
python -m probe.determinism --a results/traces/tiny-aya-global-q4km.math.jsonl --b $D/reviewer_rerun_np2_cache.jsonl \
  --label "committed (np2+cache) vs reviewer rerun (np2+cache)" --out $D/compare_committed_vs_reviewer.json
python -m probe.determinism --a results/traces/tiny-aya-global-q4km.math.jsonl --b $D/det_run1.jsonl \
  --label "committed (np2+cache) vs deterministic config" --out $D/compare_committed_vs_det.json
