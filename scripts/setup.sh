#!/usr/bin/env bash
# Fetch pinned CPU tooling into ./tools (git-ignored) and verify checksums.
set -euo pipefail
T=${TOOLS:-tools}; mkdir -p "$T/models"
LLAMA=b11815
get() { [ -f "$2" ] || curl -sL -o "$2" "$1"; echo "$3  $2" | sha256sum -c -; }
# SKIP_GENERATION=1: rescoring only needs GlotLID (~1.7 GB) and the Python deps, not llama.cpp or the 2.1 GB GGUF.
if [ "${SKIP_GENERATION:-0}" != "1" ]; then
[ -x "$T/llama-$LLAMA/llama-server" ] || { curl -sL -o "$T/llama.tgz" \
  "https://github.com/ggml-org/llama.cpp/releases/download/$LLAMA/llama-$LLAMA-bin-ubuntu-x64.tar.gz" && tar xzf "$T/llama.tgz" -C "$T"; }
get https://huggingface.co/CohereLabs/tiny-aya-global-GGUF/resolve/a602ea7eeec3a4ad6f77a1b8cf6a53512824922b/tiny-aya-global-q4_k_m.gguf \
    "$T/models/tiny-aya-global-q4_k_m.gguf" d01d995272af305b2b843efcff8a10cf9869cf53e764cb72b0e91b777484570a
fi
get https://huggingface.co/cis-lmu/glotlid/resolve/85cd6716494360367b75f642b5bc78667605d0b4/model.bin \
    "$T/glotlid.bin" a818b6bd42a628ab47d3dfc1578c7ea615c45381f3494c42535e31e8c4cafc9e
python -m pip install -q -r requirements.txt
