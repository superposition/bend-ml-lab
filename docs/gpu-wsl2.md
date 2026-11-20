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

### An irregular kernel: 1024x1024 matmul (`bench/einsum/gpu.bend`)

| mode | checksum | ms (program clock) | wall |
|---|---:|---:|---:|
| CPU parallel, no `!` | 268,622,200 | 213 | 0.3 s |
| WSL2 patched GPU, `!` depth 8 | 268,622,200 | 9,008–9,313 | 11.5–12.0 s |
| CPU fallback of the same `!` binary | 268,622,200 | 9,296 | 32.8 s |

During the GPU run: **100% GPU utilization**, 13.0 GB of managed memory in use
(the runtime reserves a large span even though the logical arrays are 12 MB).
The checksum is correct, so the kernel ran; the time is dominated by managed
page migration/page faults on an irregular access pattern, not by arithmetic.
`pow2` gets 8.4x because its pages move predictably; the matmul's heap traversal
does not.

That is the useful conclusion: **WSL2 can execute the GPU lane, but it is not a
performance path for pointer-heavy kernels.**

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
