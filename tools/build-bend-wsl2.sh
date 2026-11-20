#!/usr/bin/env bash
# Build a Bend 2.0.40 compiler from source with the WSL2 CUDA guard relaxed.
# Unsupported; see docs/gpu-wsl2.md.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
SRC="${BEND_SRC:-$HOME/bend-src-wsl2}"
PATCH="$HERE/wsl2-gpu.patch"
BUN="${BUN:-$HOME/.bun/bin/bun}"

if [ ! -d "$SRC/.git" ]; then
  git clone --depth 1 --branch v2.0.40 --filter=blob:none --sparse \
    https://github.com/bendlang/bend "$SRC"
  git -C "$SRC" sparse-checkout set bend2 guide
fi

if git -C "$SRC" apply --reverse --check "$PATCH" 2>/dev/null; then
  echo "patch already applied in $SRC"
else
  git -C "$SRC" apply "$PATCH"
  echo "applied $PATCH"
fi

echo "bun: $("$BUN" --version)"
echo "build: $BUN $SRC/bend2/main.ts <file.bend> -o <out>"
