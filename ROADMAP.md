# Roadmap — the experiment map

**Goal:** run and document every experiment below. Tracking issue:
[#18](https://github.com/superposition/bend-ml-lab/issues/18). GitHub milestone:
`Experiment map`.

A negative result is a result. An experiment closes when its measurement is
committed, not when its preferred hypothesis wins.

## Definition of done

Every ticket needs:

1. a reproducible command or script in the repo;
2. raw data (CSV/JSON) plus a short prose section in `docs/` or the README;
3. the metric, hardware, engine/build and sampling settings stated;
4. for a negative result, the result and the reason, in the ticket and in the
   docs.

The tracking issue is closed when every row is `Done`, `Negative`, or a
documented `Blocked` (external dependency).

## Completion (2026-11-20)

**12 Done, 3 Negative, 2 Blocked.** Every row has a committed measurement or a
documented external blocker; no row is silently dropped.

Highlights:

- GPU matmul: 9.0 s -> **95 ms** tree, **5.31 ms** contiguous device buffers
  ([#14](issues/14), `docs/contiguous.md`).
- The 95 ms is managed-memory latency, not tree indexing; `get_unchecked` moves
  it 4% ([#1](issues/1), `docs/plateau.md`).
- Backends agree bit-for-bit on the 64x64 matmul ([#2](issues/2),
  `docs/numerics.md`).
- CPU fork/join reaches **20.5x** at depth 10 / 32 threads ([#3](issues/3),
  `docs/cpu-scaling.md`).
- A kernel law is proved and `--verdict`-checked; a planted bug is rejected
  ([#4](issues/4), `docs/proofs.md`).
- Batched 2-layer trainer: **22 ms for 2,048 epochs**, MSE 0.000406; found and
  fixed two latent no-op kernels ([#8](issues/8), `docs/training.md`).
- WSP on a real checkpoint with a same-engine judge and a same-norm random
  control ([#10](issues/10), `docs/wsp-real.md`).
- Independent `!` calls in one parallel let: **10-14x** faster than sequential
  ([#16](issues/16), `docs/overlap.md`).

Negatives: exact softmax-sum-1 is false in F32 (#7); a tree cursor cannot beat
indexed reads (#15); parallel gradients are below the fork/join break-even at
this batch size (#9). Blocked: a second serving engine (#12) and a native
`cma=1` host (#17) are not available on this machine.

## Baseline (2026-11-19)

These are the numbers every experiment compares against. They are one run each
unless stated; see the linked docs for the full tables.

| baseline | value | source |
|---|---:|---|
| Bend native scalar matmul, `n=1024` | 3,978 ms | `bench/einsum/results.csv` |
| C `-O3` scalar, same loops | 4,048 ms | same |
| CPU parallel, 32 vCPU | 213 ms | same |
| CPU fork/join speedup | 18.5x at depth 8 | `bench/einsum/parallel.csv` |
| Patched WSL2 GPU, 2x2 tile | 95 ms (87–107) | `bench/einsum/gpu.csv` |
| Patched WSL2 GPU, 4x4 tile | 98 ms (90–102) | same |
| Scalar trainer | 3.5 ms/epoch | `train/epochs.csv` |

## Experiments

| # | experiment | area | effort | status |
|---:|---|---|---|---|
| [#1](https://github.com/superposition/bend-ml-lab/issues/1) | Explain the 95 ms plateau: tree indexing vs managed migration | runtime, GPU | S | Done |
| [#2](https://github.com/superposition/bend-ml-lab/issues/2) | Backend bit-exactness matrix (C, JS, native 1/32 threads, GPU) | numerics | S | Done |
| [#3](https://github.com/superposition/bend-ml-lab/issues/3) | Thread scaling of Bend fork/join | runtime, bench | S | Done |
| [#4](https://github.com/superposition/bend-ml-lab/issues/4) | Prove the tensor kernels in `LAWS.bend` / `PROOF.bend` | proofs | M | Done |
| [#5](https://github.com/superposition/bend-ml-lab/issues/5) | Measure the proof checker cost on kernel laws | proofs, bench | S | Done |
| [#6](https://github.com/superposition/bend-ml-lab/issues/6) | Differential fuzz of C / JS / GPU tensor kernels | numerics | M | Done (rectangular corpus) |
| [#7](https://github.com/superposition/bend-ml-lab/issues/7) | Prove softmax sums to 1 for a non-empty list | proofs | M | Negative |
| [#8](https://github.com/superposition/bend-ml-lab/issues/8) | Batched trainer: matmul path vs scalar SGD | training, bench | M | Done (MSE vs CE caveat) |
| [#9](https://github.com/superposition/bend-ml-lab/issues/9) | Parallel batch gradients with fork/join | training, runtime | L | Negative |
| [#10](https://github.com/superposition/bend-ml-lab/issues/10) | Real-checkpoint WSP analyst (export → Bend → import → judge) | WSP | L | Done |
| [#11](https://github.com/superposition/bend-ml-lab/issues/11) | Requantization noise floor for a no-op WSP edit | WSP, numerics | M | Done |
| [#12](https://github.com/superposition/bend-ml-lab/issues/12) | Same engine vs second engine: how much does the judge move a WSP score? | WSP, bench | M | Blocked (no 2nd engine) |
| [#13](https://github.com/superposition/bend-ml-lab/issues/13) | Checkpoint and resume an epoch session | training | M | Done |
| [#14](https://github.com/superposition/bend-ml-lab/issues/14) | Contiguous device tensor runtime (remove the tree for GPU) | runtime, GPU | L | Done (ceiling measured) |
| [#15](https://github.com/superposition/bend-ml-lab/issues/15) | Tree cursor / zipper for sequential traversal | runtime | L | Negative |
| [#16](https://github.com/superposition/bend-ml-lab/issues/16) | Multi-kernel overlap: can two `!` calls share the GPU? | GPU, runtime | M | Done |
| [#17](https://github.com/superposition/bend-ml-lab/issues/17) | Native-Linux GPU baseline (`cma=1`) | GPU, bench | M | Blocked (needs native host) |

Status values: `Todo`, `In progress`, `Done`, `Negative` (answer recorded, no
effect), `Blocked` (external dependency; state it in the ticket).

## Running order

1. **#1 and #2 first.** They decide whether the next large investment is the
   contiguous runtime (#14) or a native-Linux baseline (#17). #1 also tells us
   whether the 95 ms is indexing or WSL2 migration; #2 sets the numeric
   tolerance every later comparison uses.
2. **#4 and #5 next.** The proof gate is the one thing this repo can show that
   an ordinary benchmark cannot. It also forces the kernel laws to be written
   down.
3. **#8, #13, #9.** The training path: batched, resumable, then parallel.
4. **#10–#12.** The WSP path on a real checkpoint, with the noise floor and the
   judge-control established before any capability claim.
5. **#14–#16.** The runtime experiments, depending on what #1 says.
6. **#17** when a native Linux host (Jetson, VM with passthrough, or cloud) is
   reachable.

## What "done" looks like

- `docs/` contains an answered question for each row, not just a number.
- Every benchmark table names its hardware, engine and commit.
- The negative results are as visible as the positive ones.
- The 95 ms plateau is explained by a measurement, not an argument.
- At least the core tensor laws are checked by `bend PROOF.bend --verdict`.
