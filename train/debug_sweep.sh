#!/usr/bin/env bash
set -euo pipefail
export PATH="$HOME/.bend/bin:$HOME/bin:$PATH"
cd "$(dirname "$0")/.."
for e in 0 1 2 5 10 20; do
  python3 - "$e" <<'PY'
import sys
e=sys.argv[1]
s=open("train/linear_batch.bend").read().replace("train.n(40n","train.n(%sn" % e)
open("train/_lb.bend","w").write(s)
PY
  bend train/_lb.bend -o /tmp/lb >/dev/null 2>&1
  printf '%s: ' "$e"
  /tmp/lb
done
rm -f train/_lb.bend
