# Einsum in Bend 2

Bend has no tensor type and no `einsum`. What it has is a flat `Array<F32>`
(power-of-two size), `F32` arithmetic, and affine arrays. That is enough to
write the contractions machine-learning code actually uses, provided you lower
the equation yourself.

## The lowering rule

There is no runtime contraction engine in this repo, on purpose. An equation is
lowered to one specialized kernel, chosen by the index structure:

| equation | map | kernel | status |
|---|---|---|---|
| `ij,jk->ik` | `C = A B` | `tensor.matmul` | implemented, tested |
| `ji,jk->ik` | `C = A^T B` | `tensor.matmul_tn` | implemented, tested |
| `ij,kj->ik` | `C = A B^T` | needs an `mm_nt` variant | not implemented |
| `i,i->` | dot | `tensor.vdot` | implemented, tested |
| `i,j->ij` | outer | `tensor.outer` | implemented, tested |
| `ij->i` | column sums | `tensor.rowsum` | implemented, tested |
| `ij->ji` | transpose | — | not implemented |
| `bij,bjk->bik` | batched | — | not implemented |

`ij,kj->ik` is the missing one that matters for backprop (`dy * W^T`). In this
repo the toy trainer avoids it by using per-sample scalar loops; a batched
trainer would add it next. It is the same `mm` loop with B's `brow` stride = `n`
and `bcol` = 1, i.e. one more copy of `mm` — not a new idea, just another
index pattern.

Everything is row-major and `F32`. Padding to a power of two is part of the
representation, not a hidden detail: `Array.new(F32, d, 0.0)` is `2^d` slots and
the logical shape is carried beside it.

## Why not a generic einsum?

A generic engine needs, per operand, a runtime list of dimension sizes and
strides. For each output element it would:

1. run an odometer over the output labels,
2. run an odometer over the contracted labels,
3. compute each input's flat offset from its rank and strides,
4. call `Array.get`, which descends a balanced tree (`O(log n)`).

Steps 1 and 3 are rank-dependent control flow over `List<Nat>`, which Bend
supports but pays for. The specialized kernels hoist all of that into the
function signature and unroll nothing — they just hard-code the strides.

`matmul_tn` is the cheapest possible demonstration of this cost: it differs from
`matmul` only in the A index pattern (`row` instead of `row*k`, step `m` instead
of 1). At `n = 1024` it is **6.7 s vs 4.0 s**, a 1.7x penalty for one strided
read. A generic rank-N version stacks that penalty per operand and adds the
odometer.

So the practical architecture is:

```
equation text -> parse -> choose kernel + shapes -> emit a Bend call
```

A small Python code generator can do the parse/choose step today; the emitted
call is a normal Bend function. That is the natural next commit, not a runtime
interpreter.

## Measured, `n = 1024`, 32 vCPU, Bend 2.0.40

| case | checksum | ms |
|---|---:|---:|
| generation (two 1M-entry vectors) | 1,048,938.6 | 11 |
| `matmul` scalar | 268,622,200 | 3978 |
| `matmul` 256 lanes (depth 8) | 268,622,200 | 213 |
| `matmul_tn` scalar | 268,641,470 | 6682 |
| `outer` (1M entries) | 263,967.7 | 3 |
| `vdot` (1M entries) | 262,327.66 | 6 |

Cross-language, same loops and checksums:

| implementation | ms |
|---|---:|
| NumPy BLAS (reference) | 24.1 |
| Bend 256 lanes | 213 |
| C + OpenMP (32 threads) | 223 |
| Bend scalar | 3978 |
| C `-O3` scalar | 4048 |
| Node `Float32Array` | 4503 |
| Bend JS target | 29,087 |

Reproduce:

```bash
cd bench/einsum
bash run.sh          # writes results.csv
bash sweep.sh        # depth 0..8 at n=1024
```

The C/Node/NumPy twins use the same xorshift32 generator and the same `i,j,k`
loop order. Checksums agree to ~3e-8 relative; the last-bit difference is float
accumulation order, not a logic difference.

## The GPU lane

`!` on a `matmul` call hands the fork tree to the GPU; the same `tree` function
is the kernel. On a native Linux host with CUDA this is the intended path.

WSL2 reports `concurrentManagedAccess=0`, and Bend 2.0.40 refuses to start the
GPU lane. With the guard relaxed (`docs/gpu-wsl2.md`) the GPU executes and
produces the correct checksum, but the result is not a performance number:

| mode | ms | note |
|---|---:|---|
| CPU parallel, no `!` | 213 | the 32-vCPU path |
| WSL2 patched GPU, `!` depth 8 | 9,008–9,313 | 100% util, 13 GB managed |
| `pow2(20)`, CPU | 21,920 | regular access |
| `pow2(20)`, patched GPU | 2,600 | 8.4x; regular access |

The irregular matmul heap walk page-faults managed memory over PCIe; the
regular `pow2` tree does not. Treat the WSL2 lane as correctness-only. Native
Linux, a GPU-passthrough VM, or the Jetson is required for a meaningful GPU
benchmark.

## Correctness gates used here

- Every kernel is compared to a NumPy reference on small cases
  (`tests/tensor_test.bend` prints 4x4 results; see the commit history for the
  matching array).
- Every benchmark prints a checksum; a speedup with a different checksum is a
  bug, not a result.
- Loop order and generator are identical across languages.
- `bend --check-only` runs the type checker (including termination and
  quantities) on each file.
