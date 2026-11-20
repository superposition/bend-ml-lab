#!/usr/bin/env bash
# Parallel scaling at n=1024. Writes depth,ms rows.
set -euo pipefail
export PATH="$HOME/.bend/bin:$HOME/bin:$PATH"
cd "$(dirname "$0")"
echo "depth,lanes,checksum,ms" > parallel.csv
for d in 0 2 4 6 8; do
  python3 - "$d" <<'PY'
import sys
d = sys.argv[1]
s = open("pmm.bend").read().replace("compute(8n)", "compute(%sn)" % d)
open("_pmm.bend", "w").write(s)
PY
  bend _pmm.bend -o /tmp/pmm >/dev/null
  out=$(/tmp/pmm)
  cs=$(printf '%s' "$out" | cut -d, -f1)
  ms=$(printf '%s' "$out" | cut -d, -f2)
  lanes=$(( 1 << d ))
  printf '%s,%s,%s,%s\n' "$d" "$lanes" "$cs" "$ms" | tee -a parallel.csv
done
rm -f _pmm.bend
