# Tree cursor: negative

Ticket #15 asked whether a zipper/cursor over `ALeaf`/`ANode` can beat random
`Array.get` without a runtime change. The measurements say no, before writing
one.

The cursor's best case is a structural walk: visit each node once, no index
math. #1 measured exactly that on 1M `F32` values:

| path | CPU ms | GPU ms |
|---|---:|---:|
| indexed `Array.get` | 1 | 176 |
| structural fold (the cursor's access pattern) | 15 | 5,412 |
| linked list | 3 | 2,246 |

A cursor adds a pending-sibling stack on top of the structural walk, so 15 ms
CPU / 5.4 s GPU is a **lower bound** for it, not an estimate. It is 15x slower
than the indexed path on CPU and 31x on the GPU. #14 then showed the real lever:
device buffers are 5.31 ms end to end vs 95 ms for the tree.

Conclusion: **negative**. A cursor is an interesting data structure but not a
performance path for this workload. The remaining work is memory ownership
(device buffers), not traversal order.
