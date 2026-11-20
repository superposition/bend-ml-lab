# Running Bend's GPU lane on WSL2

WSL2 exposes the RTX 4090 to CUDA, but it does **not** expose *concurrent
managed access*. Bend 2.0.40 refuses to start its GPU lane when that attribute
is 0. This note records why, and the only workaround that has been measured
here.

## The check

`bend2/comp.ts`, `gpu_probe()`:

```c
GPU_CHECK(cuDeviceGetAttribute, &managed,
  CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS, gpu_dev);
if (managed == 0) {
  return "CUDA device lacks concurrent managed access (WSL2 lacks it)";
}
```

A direct CUDA probe on this machine (CUDA 12.8, driver exposed by WSL2):

```
devices=1
0 NVIDIA GeForce RTX 4090 cc=8.9 cma=0 unifiedAddr=1 managed=1
```

`managed=1` and `unifiedAddr=1`: managed allocations and unified addressing
work. `cma=0`: the host may not touch managed memory *while a kernel is
running*. There is no registry key, driver option or `nvidia-smi` setting that
turns `cma` on; it is a property of the WSL2 GPU paravirtualization. Docker
Desktop inherits it.

## Why the guard is conservative

Bend's CUDA runtime uses one managed heap:

```c
cuMemAllocManaged(&p, bytes, CU_MEM_ATTACH_GLOBAL);
cuMemAdvise(p, bytes, CU_MEM_ADVISE_SET_PREFERRED_LOCATION, gpu_dev);
...
```

and runs a pass as:

```c
cuMemcpyHtoD(gpu_front, &H[H_FRONT], 8);
for (...) {
  gpu_run();                 // launch kernels
  cuCtxSynchronize();        // wait
}
```

The host reads `H[H_FRONT]` only **after** `cuCtxSynchronize()`. No host access
overlaps a running kernel, which is exactly the operation `cma` protects. The
runtime is therefore safe without `cma` as long as this pass structure holds;
the check is a conservative platform gate, not a memory-model requirement.

## The patch (unsupported)

`tools/wsl2-gpu.patch` relaxes the guard to a warning:

```diff
   if (managed == 0) {
-    return "CUDA device lacks concurrent managed access (WSL2 lacks it)";
+    fputs("bend: warning: no concurrent managed access; continuing", stderr);
   }
```

Build a compiler from the v2.0.40 tag and run it with Bun (the release binary
is already compiled and cannot be patched this way):

```bash
bash tools/build-bend-wsl2.sh
# clones ~/bend-src-wsl2, applies the patch, prints the bun command

~/.bun/bin/bun ~/bend-src-wsl2/bend2/main.ts bench/einsum/gpu.bend -o /tmp/mm_gpu
/tmp/mm_gpu +RTS --gpu 4GB
```

This is **not** a supported configuration. It is a correctness experiment: the
same generated kernels, the same checksums, one warning. Do not report its
numbers as WSL2 GPU performance.

## What was measured

### A regular kernel: `pow2(20)`

`pow2` splits a balanced tree; the work is regular and the managed pages are
touched in order.

| mode | wall |
|---|---:|
| CPU (32 vCPU) | 21.92 s |
| GPU (patched, first run) | 3.41 s |
| GPU (patched, warm) | 2.60 s |

The GPU lane is real: ~8.4x over the CPU. `nvidia-smi` shows the device active
during the run.

### An irregular kernel: 1024x1024 matmul

The first working version was 9.0 s. The final one is 95 ms. These are the
steps that mattered, all with the same checksum (`268622200`) and the same
generated kernels:

| version | lanes | ms | what changed |
|---|---:|---:|---|
| naive, 256 lanes | 256 | 9,008 | one dot per output cell; column stride N |
| transposed B, 256 lanes | 256 | 1,373 | both inner-loop operands advance by 1 |
| 1D tree, depth 10 | 1,024 | 470 | one output row per lane |
| 2D tree, 4096 lanes | 4,096 | 230 | 1 row x 256 cols per lane |
| 2x2 tiled, 2048 lanes | 2,048 | 141 | 2 rows x 256 cols; 4 reads per 4 MACs |
| **2x2 tiled, 1024 lanes** | **1,024** | **95** (87-107) | 2 rows x 512 cols; `--gpu 512MB` |
| CPU parallel (reference) | 32 vCPU | 213 | no `!` |

Why this order:

1. **Layout first.** A column stride of N makes every inner-loop read a tree
   walk into a different part of the heap. Transposing B on the host is a
   12 ms CPU pass and cuts the GPU time 6.5x.
2. **Occupancy second.** One lane per output cell is 1M lanes, more than the
   16,384-task cube. One row per lane (1,024) is the first useful split; the
   remaining gain comes from splitting columns as well, then stopping at 1,024
   lanes. 4,096 lanes is slower: the per-lane work gets too small.
3. **Reuse third.** A 2x2 tile reads four values (two A, two B) per four
   multiply-adds instead of eight. That is the 230 -> 141 ms step.
4. **Heap size last.** `--gpu 512MB` is the smallest span that fits 1,024
   lanes; it is ~1.5x faster than 4 GB because the managed heap has fewer pages
   to track/migrate.

The final configuration is in `bench/einsum/gpu_tile.bend`:

```bash
~/.bun/bin/bun ~/bend-src-wsl2/bend2/main.ts bench/einsum/gpu_tile.bend -o /tmp/tile
/tmp/tile +RTS --gpu 512MB
```

Program-clock kernel: **87-107 ms** (median 95 ms) against **213 ms** for the
32-vCPU parallel build, so the patched GPU is **2.2x faster on compute**.
Wall-clock for one process is **0.69 s vs 0.3 s** because GPU setup is ~0.5 s;
a session of several matmuls amortizes it. The first version took 9.0 s, so the
optimization is **99x** end to end.

The GPU is genuinely active (100% utilization, 13 GB managed at 4 GB heap; less
at 512 MB) and the checksum is identical. The remaining gap to a native CUDA
BLAS is the tree layout: every value is a separately allocated node, so the
kernel is latency-bound on pointer chasing, not on 4090 FLOPs. A contiguous
device buffer would need a runtime change, not a Bend program.

## What to use for real GPU numbers

1. **Native Linux on the same box.** The 4090 supports `cma=1` with the normal
   driver; that is the intended Bend target. This is the only option that gives
   a number comparable to the published Metal/CUDA results.
2. **A Linux VM with the GPU passed through** (VFIO/PCIe passthrough or Windows
   DDA). The guest gets the real driver and `cma`; WSL2 does not.
3. **The Jetson board** (`ssh pinkie`, Orin NX, native Linux, unified memory)
   when it is powered and reachable. The Bend release ships `linux-arm64`; CUDA
   is part of JetPack. It is slower than the 4090 but architecturally correct.
4. **A cloud GPU** (L4/A10/4090 instance) with native Linux and CUDA 12/13.

Cloud and bare metal also avoid the 13 GB managed reservation observed here,
which is a WSL2-specific overhead of the current runtime.

## If "concurrent" meant CPU

Bend already has CPU concurrency: `a b = f(x) g(x)` forks, `Array.fork` gives
lanes disjoint handles into one array, and `Array.join` merges them. That is what
`src/parallel.bend` uses to get 18.5x on 32 vCPU. `!` is the GPU marker, not the
CPU one; removing it keeps the CPU fork tree.

For a concurrent trainer, the missing piece is a **batch gradient**: fork the
weights read-only, split the 256 samples into lanes, compute a partial gradient
per lane, then reduce. The forward/backward scalar loops in `src/mlp.bend` would
become a `tree` over samples with a list/array reduction at the join. That is a
larger change than the GPU patch and is not implemented in this repo yet.
