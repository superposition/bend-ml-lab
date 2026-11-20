#!/usr/bin/env bash
# Thread scaling of the 1D fork/join matmul, n=1024.
set -euo pipefail
export PATH="$HOME/.bend/bin:$HOME/bin:$PATH"
cd "$(dirname "$0")"
echo "depth,threads,rep,checksum,ms" > threads.csv
for d in 0 2 4 6 8 10; do
  python3 - "$d" <<'PY'
import sys
d = sys.argv[1]
s = open("pmm.bend").read().replace("compute(8n)", "compute(%sn)" % d)
open("_pmm.bend", "w").write(s)
PY
  bend _pmm.bend -o /tmp/pmm_d >/dev/null
  for t in 1 2 4 8 16 32; do
    for r in 1 2 3; do
      out=$(/tmp/pmm_d +RTS --threads "$t")
      cs=${out%,*}; ms=${out#*,}
      printf '%s,%s,%s,%s,%s\n' "$d" "$t" "$r" "$cs" "$ms" | tee -a threads.csv
    done
  done
done
rm -f _pmm.bend
