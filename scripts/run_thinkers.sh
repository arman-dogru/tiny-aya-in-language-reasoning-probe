#!/usr/bin/env bash
# NOT YET RUN. The intended primary comparison: Tiny Aya L2-Thinker vs En-Thinker.
# Both checkpoints are gated on Hugging Face ("gated: auto" -> accept the licence
# while logged in, then use a read token). No HF token was available on the build
# box, so this script is untested; it is the exact path to swap the substitute out.
#
#   export HF_TOKEN=...            # after accepting the gate on both model pages
#   bash scripts/run_thinkers.sh
set -euo pipefail
: "${HF_TOKEN:?accept the gate on CohereLabs/tiny-aya-l2-thinker and -en-thinker, then export HF_TOKEN}"
T=${TOOLS:-tools}
[ -d "$T/llama.cpp" ] || git clone --depth 1 --branch b11815 https://github.com/ggml-org/llama.cpp "$T/llama.cpp"
python -m pip install -q -r "$T/llama.cpp/requirements/requirements-convert_hf_to_gguf.txt"
for v in l2 en; do
  name=tiny-aya-$v-thinker
  [ -f "$T/models/$name-q4_k_m.gguf" ] && continue
  hf download "CohereLabs/$name" --local-dir "$T/hf/$name"
  python "$T/llama.cpp/convert_hf_to_gguf.py" "$T/hf/$name" --outtype bf16 --outfile "$T/models/$name-bf16.gguf"
  "$T/llama-b11815/llama-quantize" "$T/models/$name-bf16.gguf" "$T/models/$name-q4_k_m.gguf" Q4_K_M
done
# The thinkers' chat template already prepends "Think in the same language as the prompt."
# and opens <|START_THINKING|>, so the "native" arm is the faithful one; --think-open ''.
for v in l2 en; do
  name=tiny-aya-$v-thinker; port=$([ $v = l2 ] && echo 8093 || echo 8094)
  "$T/llama-b11815/llama-server" -m "$T/models/$name-q4_k_m.gguf" -t 1 -tb 2 -c 8192 --port $port -np 2 \
    --swa-full --jinja > "$T/server-$name.log" 2>&1 &
done
sleep 30
for v in l2 en; do
  name=tiny-aya-$v-thinker; port=$([ $v = l2 ] && echo 8093 || echo 8094)
  python -m probe.run --server http://127.0.0.1:$port --model "$name-q4km" --think-tags --think-open '' \
    --think-close '<|END_THINKING|>' --arms native --max-think 2048 --workers 2 \
    --out "results/traces/$name-q4km.jsonl" &
done
wait %3 %4
echo "score with: python -m probe.score --run results/traces/tiny-aya-l2-thinker-q4km.jsonl:aya --run results/traces/tiny-aya-en-thinker-q4km.jsonl:aya ..."
