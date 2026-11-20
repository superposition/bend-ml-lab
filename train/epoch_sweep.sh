#!/usr/bin/env bash
# Runs the trainer for 1,2,4,...,64 epochs and writes epochs,loss,total_ms.
set -euo pipefail
export PATH="$HOME/.bend/bin:$HOME/bin:$PATH"
cd "$(dirname "$0")"
echo "epochs,final_loss,total_ms" | tee epochs.csv
for e in 1 2 4 8 16 32 64; do
  python3 - "$e" <<'PY'
import sys
e = sys.argv[1]
s = open("main.bend").read().replace("run(20n, 20n", "run(%sn, %sn" % (e, e))
open("_train.bend", "w").write(s)
PY
  bend _train.bend -o /tmp/train_sweep >/dev/null
  start=$(date +%s%N)
  /tmp/train_sweep | tail -1 > /tmp/last.txt
  end=$(date +%s%N)
  ms=$(( (end - start) / 1000000 ))
  loss=$(cut -d, -f2 /tmp/last.txt)
  printf '%s,%s,%s\n' "$e" "$loss" "$ms" | tee -a epochs.csv
done
rm -f _train.bend
