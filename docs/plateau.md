# Why the GPU matmul is 95 ms

Ticket #1 asked whether the 95 ms plateau is **tree indexing** or **WSL2
managed-memory migration**. The answer is neither of the things a tree cursor or
`Array.get_unchecked` would fix: it is per-value managed-memory latency.

## Method

One million `F32` values, four read paths, fresh input per path, generation
outside the timed region:

- **indexed** — `Array.get` in a loop (`T.checksum`);
- **structural** — recurse over `ALeaf`/`ANode` with no index math (`T.sum_tree`);
- **unchecked** — `Array.get_unchecked` (`P.csumu`);
- **list** — `Array.to_list` on the CPU, then fold the linked list.

CPU is the native build (installed `bend`, no `!`); GPU is the patched WSL2
compiler with `+RTS --gpu 512MB`.

## Result

| path | CPU ms | GPU ms |
|---|---:|---:|
| indexed | 1 | 176 |
| unchecked | 1 | 169 |
| structural | 15 | 5,412 |
| list | 3 | 2,246 |

## Reading

- **Index math is not the cost.** Removing the size/bounds work
  (`get_unchecked`) moves the GPU from 176 to 169 ms — a 4% change. The tree
  walk itself is not what costs 95 ms in the matmul.
- **Structural recursion is terrible on the GPU.** 5.4 s for a 1M-value fold:
  one function call per node does not map to GPU lanes.
- **Linked lists are worse.** 2.2 s, 12.8x the indexed path: pointer chasing
  managed memory is the worst case.
- **CPU has no problem.** The indexed fold is 1 ms, 176x faster than the GPU
  for the same work. The CPU's caches hide the tree.

So the matmul's 95 ms is dominated by **managed-memory access latency on the
GPU lane**, not by `Array.get`. A tree cursor (#15) or `get_unchecked` will not
move it. The levers are contiguous device memory (#14) or a native Linux host
with `cma=1` (#17).

Raw data: [`bench/einsum/fold.csv`](../bench/einsum/fold.csv).
