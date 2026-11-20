#!/usr/bin/env bash
# Runs the einsum family at n=1024 and writes bench/einsum/results.csv.
# Requires: bend on PATH, gcc, node, and a Python with numpy.
set -euo pipefail
export PATH="$HOME/.bend/bin:$HOME/bin:$PATH"
cd "$(dirname "$0")"
N="${N:-1024}"
PY="${PY:-python3}"
OUT=results.csv

echo "case,lang,mode,checksum,ms" > "$OUT"

BEND="${BEND:-bend}"
"$BEND" main.bend -o /tmp/bend_einsum >/dev/null
/tmp/bend_einsum | awk -F, -v OFS=, '{print $1,"Bend","native",$2,$3}' >> "$OUT"

gcc -O3 -o /tmp/mm_seq c/main.c
/tmp/mm_seq "$N" | awk -F, -v OFS=, '{print $3,"C",$2,$4,$5}' >> "$OUT"

gcc -O3 -fopenmp -o /tmp/mm_par c/main.c -DPAR
/tmp/mm_par "$N" | awk -F, -v OFS=, '{print $3,"C+OpenMP",$2,$4,$5}' >> "$OUT"

node node/main.js "$N" | awk -F, -v OFS=, '{print $3,"Node",$2,$4,$5}' >> "$OUT"

"$PY" py/main.py "$N" | awk -F, -v OFS=, '{print $3,"NumPy-BLAS",$2,$4,$5}' >> "$OUT"

cat "$OUT"
