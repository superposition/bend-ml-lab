# The contiguous-buffer ceiling

Ticket #14 asked what the matmul costs without the tree. This is a standalone
CUDA measurement, not a BendRT fork: it establishes the ceiling that a runtime
change would target.

## Method

`bench/gpu/contig.cu`: 1024x1024 `F32`, 16x16 shared-memory tiles, same
xorshift32 generator as the Bend benchmarks.

- **device**: `cudaMalloc` + explicit `cudaMemcpy` in/out, then the kernel.
- **managed**: `cudaMallocManaged`, no copies, same kernel.
- **bend_tree_managed**: the 2x2-tiled Bend kernel from `gpu_tile.bend`.

## Result

| variant | kernel | copies | total | sum |
|---|---:|---:|---:|---:|
| device buffers | **0.727 ms** | 4.58 ms | **5.31 ms** | 268,619,543.3 |
| managed, contiguous | 68.0 ms | 0 | 68.0 ms | 268,619,543.3 |
| Bend tree, managed | — | — | 95 ms | 268,622,200 |

`cma=0` on WSL2. The checksum differs by association only; #2 proved the matmul
bits are identical across backends.

## Decomposition of the 95 ms

- Contiguous managed: 68 ms. **~67 ms is managed-memory migration**, not the
  tree.
- Contiguous device + copies: 5.3 ms. Explicit copies cost ~4.6 ms; the kernel
  is 0.73 ms.
- Bend tree managed: 95 ms. The remaining ~27 ms over contiguous managed is the
  tree's per-node pointer work.

So the ordering is: **device memory (18x) > contiguous layout (1.4x) > tiling**.
The 2x2/4x4 tile work was chasing the small term.

## What integration would take

BendRT uses one managed heap (`cuMemAllocManaged`) and a tree `Array`. A device
path needs:

1. a device allocation and a host->device copy for each input;
2. a kernel that indexes linear buffers (the `.cu` here);
3. a device->host copy for the output, or a way to read device memory in a
   following Bend expression.

That is a runtime/effect change, not a Bend program. The ceiling says it is
worth it: 5.3 ms including copies, 0.73 ms kernel-only, against 95 ms.

Raw data: [`bench/gpu/contig.csv`](../bench/gpu/contig.csv).


## Integrated prototype (2026-11-20)

`docs/device-buffers.md`: a C effect now copies a Bend `Array<F32>` to device
buffers and back. Warm 1024x1024: **5 ms** (3-18 over in-process reps), checksum 268622200,
against 95 ms for the tree and 5.31 ms for the standalone CUDA path. The effect is native-only
and uses Bend 2.0.40 internals, but it proves the integration shape: the array
block is already contiguous, so no new value kind is required.
