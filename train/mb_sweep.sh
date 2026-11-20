#!/usr/bin/env bash
set -euo pipefail
export PATH="$HOME/.bend/bin:$HOME/bin:$PATH"
cd "$(dirname "$0")/.."
for e in 0 1 2 5; do
  python3 - "$e" <<'PY'
import sys
e=sys.argv[1]
s=open("train/mlp_batch.bend").read().replace("train.n(64n","train.n(%sn" % e)
open("train/_mb.bend","w").write(s)
PY
  bend train/_mb.bend -o /tmp/mb >/dev/null 2>&1
  printf '%s: ' "$e"; /tmp/mb
done
rm -f train/_mb.bend
