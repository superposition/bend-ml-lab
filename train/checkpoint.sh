#!/usr/bin/env bash
# Save at epoch 5, resume to epoch 10, compare with the in-process reference.
set -euo pipefail
export PATH="$HOME/.bend/bin:$HOME/bin:$PATH"
cd "$(dirname "$0")/.."
SAVE=$(bend train/save.bend)
RES=$(bend train/resume.bend)
echo "$SAVE"
echo "$RES"
REF=$(printf '%s\n' "$SAVE" | cut -d, -f3)
GOT=$(printf '%s\n' "$RES" | cut -d, -f3)
echo "reference_loss,resume_loss" > train/checkpoint.csv
echo "$REF,$GOT" >> train/checkpoint.csv
python3 - "$REF" "$GOT" <<'PY'
import sys
a=float(sys.argv[1]); b=float(sys.argv[2])
print("match" if abs(a-b) <= 1e-9 else "DRIFT %.3e" % abs(a-b))
PY
