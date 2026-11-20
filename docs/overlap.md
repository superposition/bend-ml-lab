# Multi-kernel overlap

Ticket #16 asked whether two independent `!` calls share the GPU.

## Method

Two independent 512x512 matmuls (2x2-tiled, `rd=8, cd=1`), each with its own
inputs. Two forms:

- **sequential**: two `pmatmul3_bt!` calls in a row;
- **parallel let**: `l r = pmatmul3_bt!(...) pmatmul3_bt!(...)`.

Both variants include the two checksums in the timed region. GPU is the patched
WSL2 compiler, `+RTS --gpu 512MB`.

## Result

| mode | ms | sum |
|---|---:|---:|
| CPU sequential | 11 | 67,218,400 |
| CPU parallel let | 9 | 67,218,400 |
| GPU sequential | 123–163 | 67,218,400 |
| **GPU parallel let** | **11–12** | 67,218,400 |

## Reading

- On the GPU, the parallel let is **10–14x faster** than two sequential calls.
  The two calls share one GPU pass/heap setup instead of paying the ~60 ms
  setup twice. This is amortization more than concurrent kernels, but the
  effect is what a caller cares about.
- The emitted sums are identical, so both calls ran and produced the same
  results.
- On the CPU the difference is small (11 vs 9 ms): the runtime already
  parallelizes inside each call, and setup is cheap.

Practical rule: when a workload has several independent GPU operations, put
them in one parallel let rather than calling them one after another. The runtime
will fuse the pass.

Raw: [`bench/einsum/overlap.csv`](../bench/einsum/overlap.csv).
