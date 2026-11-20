# bend-ml-lab

[![ci](https://github.com/superposition/bend-ml-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/superposition/bend-ml-lab/actions/workflows/ci.yml)

Measured experiments in **Bend 2**: einsum kernels, training epochs, and
weight-space diagnostics. The goal is to find out what this language is good
for on numeric work by building the same thing three ways and timing it, not by
reading the landing page.

- Bend **2.0.40** (`bendlang/bend`), Apache-2.0.
- Verified **2026-11-19** on WSL2 Ubuntu 22.04, 32 vCPU, RTX 4090 (CUDA 12.8),
  clang 21.1.8, Node 12, NumPy 2.2.6.
- Numbers below are one run each from `bench/einsum/run.sh`; repeated Bend and
  C runs varied by up to ~10%. Scripts, checksums and raw CSV are in the repo.

Bend is roughly ten weeks old as a 2.x line. Everything here is a measurement
of *this* version, not a claim about where the language will be next month.

**Roadmap:** [`ROADMAP.md`](ROADMAP.md) · tracking issue
[#18](https://github.com/superposition/bend-ml-lab/issues/18) · milestone
`Experiment map`. A negative result is a result.

## What is here

| Track | Question | Entry point |
|---|---|---|
| Einsum | How do you express `ij,jk->ik` and friends, and what does it cost? | `bench/einsum/` |
| Epochs | What does a training session cost per epoch, and how does it scale? | `train/` |
| WSP | Can Bend act as the analyst in a weight-space-programming loop? | `wsp/` |
| Notes | The language traps that cost real time | `docs/bend-notes.md` |

## Results

### 1. Matmul, `n = 1024` (1.07 GFLOP), row-major `F32`

| implementation | ms | note |
|---|---:|---|
| NumPy `A @ B` | 24.1 | BLAS, multithreaded; optimized reference |
| Bend native, 256 lanes | 213 | `Array.fork` tree, depth 8 |
| C + OpenMP, 32 threads | 223 | same `i,j,k` loops |
| Bend native, scalar | 3978 | same loops |
| C `-O3`, scalar | 4048 | same loops |
| Node 12, `Float32Array` | 4503 | same loops |
| Bend JS target, scalar | 29,087 | `bend file.bend` interpretation path |

Bend's native scalar code is **within 0.4% of `gcc -O3`** on this kernel, and its
fork/join version is **within 4% of OpenMP**. The JS target is ~7x slower and
ignores `!` entirely.

Checksums agree to ~3e-8 relative: Bend `268622200`, C/NumPy `268622208`; the
difference is float accumulation order / FMA contraction, not a logic error.
The checker and the runtime agree with each other.

**GPU on WSL2 (patched, unsupported):** the 4090 reports
`concurrentManagedAccess=0`, so Bend refuses the GPU lane by default. With the
guard relaxed (`docs/gpu-wsl2.md`), the tiled 2x2 kernel runs the 1024 matmul in
**87–107 ms** (median 95 ms) against **213 ms** on 32-vCPU parallel — **2.2x
faster on compute**, 99x faster than the first working GPU version (9.0 s).
A C effect over Bend arrays now copies to real device buffers: **16 ms warm**
(195 ms cold, context setup) with the same checksum, vs 95 ms for the tree.
Single-process wall for the tree path is 0.69 s vs 0.3 s because GPU setup is
~0.5 s; repeated matmuls amortize it. A 4x4 tiled variant halves reads again but needs four
times fewer lanes and lands at ~98 ms — a tie, which is why 2x2/1,024 lanes is
the recommended config. Native Linux remains the supported path.

### 2. Parallel scaling (`n = 1024`, Bend native)

| fork depth | lanes | ms | speedup |
|---:|---:|---:|---:|
| 0 | 1 | 3970 | 1.0x |
| 2 | 4 | 3969 | 1.0x |
| 4 | 16 | 954 | 4.2x |
| 6 | 64 | 316 | 12.6x |
| 8 | 256 | 215 | 18.5x |

Depth 2 (4 lanes) gives **no speedup at all** here: fork/join overhead cancels the split. The scheduler
wants balanced, coarse chunks; 256 lanes on 32 vCPU works because the rows are
uniform.

### 3. Epoch breakdown (2-2-8-4 MLP, 256 samples, per-sample SGD)

| epochs | final cross-entropy | total ms | ms/epoch |
|---:|---:|---:|---:|
| 1 | 0.2537 | 6 | 6.0 |
| 2 | 0.0883 | 9 | 4.5 |
| 4 | 0.0293 | 16 | 4.0 |
| 8 | 0.00962 | 30 | 3.8 |
| 16 | 0.00329 | 57 | 3.6 |
| 32 | 0.00122 | 114 | 3.6 |
| 64 | 0.000487 | 222 | 3.5 |

Linear in epochs once the first-epoch allocation cost is paid. The first epoch
pays for list allocation and warm cache; after that the cost is flat.

### 4. WSP diagnostics (same model, after 20 epochs)

| probe | mean loss | delta vs baseline |
|---|---:|---:|
| baseline | 0.002368 | — |
| scale `W2` row 0 (class 0) by 1.1 | 0.002279 | −3.8% |
| scale row 1 | 0.002240 | −5.4% |
| scale row 2 | 0.001996 | −15.7% |
| scale row 3 | 0.002193 | −7.4% |
| add a random `W2` direction, ε = 0.05 | 0.002451 | +3.5% |

`W2` after training: mean 0.0221, std 0.9514, rms 0.9517, min −1.5616, max
1.7637. These are directional sensitivities on a toy model. They are **not** a
capability result and not a replication of any published WSP edit. See
[`docs/wsp.md`](docs/wsp.md) for the evaluation rules this repo follows.

## Findings

1. **Native Bend is C-class on dense `F32` loops.** Same loop order, same
   checksum, 0.4% apart. The claim survives contact with a naive kernel.
2. **The JS target is for development only.** It is one core, it ignores `!`,
   and it is ~7x slower here. Do not measure in JS.
3. **Fork/join scales, with a floor.** 18.5x on 32 vCPU at depth 8; depth 2 (4 lanes)
   gives no gain because the chunks are too small.
4. **Affine arrays are the language's main ergonomic cost.** `Array<T>` is a
   `Type`, not `Data`: reads return the array, writes rebind it, and you cannot
   share one across a parallel call without `Array.fork` (`@unsafe`) and
   disjoint write bands. This shapes `src/tensor.bend` and is why the parallel
   kernels live in `src/parallel.bend`.
5. **Einsum is a lowering problem, not a runtime problem.** The practical
   design is a parser that lowers an equation to a small set of specialized
   kernels. A generic N-D strided contraction would pay O(rank) index math and
   O(log n) array access per element; `mm_tn` already shows the cost of strided
   reads (6.7 s vs 4.0 s).
6. **A `Data` model of lists is a good fit for small trainings.** Lists are
   copyable, so weight updates are pure list rebuilds with no ownership
   puzzles. The price is O(n) list indexing; at 60 weights it is invisible
   (3.5 ms/epoch).
7. **WSL2's GPU lane needs a patch, and layout dominates.** The 4090 reports
   `concurrentManagedAccess=0`; Bend's guard is conservative, so relaxing it
   executes correctly. The first matmul took 9.0 s; transposing B, using a 2D
   fork tree, 2x2 tiling and a 512 MB heap brought it to 95 ms — 2.2x faster
   than 32-vCPU parallel. The rest is the tree layout: every value is a heap
   node, so the kernel is pointer-latency-bound, not FLOP-bound.
8. **No Windows target.** WSL works; native Windows does not. CUDA/Metal need a
   native host anyway.
9. **No laws yet.** `bend --check-only` passes on the safe core (`ALL PROOFS
   CHECK`), but that only means types, quantities and termination check; there is
   no `LAWS.bend` here. Bend's differentiator is not exercised yet. The next
   step is to state kernel laws (bounds, transpose identities, `lset`/`lget`
   round-trip) and prove them with `PROOF.bend`.

## Quickstart (WSL2)

```bash
# once
curl -fsSL https://bend-lang.com/install.sh | sh
export PATH="$HOME/.bend/bin:$HOME/bin:$PATH"
ln -sf /usr/bin/clang-21 "$HOME/bin/clang"   # any clang 14+; 19+ for `!`

cd bend-ml-lab
bend src/tensor.bend --check-only            # ALL PROOFS CHECK (no laws yet)
bend src/parallel.bend --check-only          # reports the 2 @unsafe defs, by design
bend train/main.bend                         # print per-epoch CSV
bend wsp/analyze.bend                        # print the probe table
```

Native runs:

```bash
bend bench/einsum/main.bend -o /tmp/bend_einsum
/tmp/bend_einsum
```

`bench/einsum/run.sh` builds the Bend binary and the C/Node/NumPy twins and
writes `bench/einsum/results.csv`.

## Repo map

```
src/tensor.bend       safe core: flat F32 arrays, dot/mm/mm_tn, outer, vdot, rowsum
src/parallel.bend     @unsafe fork/join tree (pmatmul); the only unsafe module
src/mlp.bend          Model (lists), forward, backprop, per-sample SGD, eval
bench/einsum/         Bend benchmark + C, Node, NumPy twins + runner + sweep
train/main.bend       per-epoch loss/time CSV
train/epoch_sweep.sh  1..64 epoch scaling table
wsp/analyze.bend      stats, row scaling, random-direction control
bench/einsum/gpu.bend GPU probe (`!`), built with the patched WSL2 compiler
bench/einsum/gpu_tile.bend final 2x2-tiled GPU matmul (95 ms, 1024 lanes)
tools/wsl2-gpu.patch   relax the WSL2 CUDA guard (unsupported; docs/gpu-wsl2.md)
docs/gpu-wsl2.md      why cma=0 on WSL2, the patch, and the measured result
docs/bend-notes.md    language idioms and traps found while building this
docs/einsum.md        equation -> kernel table and the generic-engine argument
docs/training.md      model, gradients, epoch costs, checkpoint options
docs/wsp.md           analyst-vs-judge protocol and the integrity traps
ROADMAP.md            the 17 experiments, acceptance rules, running order
```

## Scope and honesty

Bend 2 is new, the compiler is largely AI-written, and the checker has no proof
of its own (`--verdict` uses a Lean-checked kernel). This repo therefore:

- measures the engine that is actually installed (`bend version` in every run);
- keeps checksums and the exact loop order, so a faster result cannot come from
  doing less work;
- states single-machine numbers, with the hardware in the table header;
- does not compare against published Bend numbers from another machine;
- does not claim a WSP capability gain.

## License

Apache-2.0. See [`LICENSE`](LICENSE). Bend itself is Apache-2.0.
