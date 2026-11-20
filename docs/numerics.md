# Backend bit-exactness

Ticket #2 asked whether C, JS, native CPU (1 and 32 threads) and the GPU lane
produce the same bits. On the cases tested, they do.

## Method

64x64 `F32` matmul, deterministic xorshift32 inputs, FNV-1a over the raw
`F32.bits` of the 4096 outputs. Two value ranges:

- `positive`: `[0,1)`;
- `negative`: `[-1,1)`.

Every backend uses the same generator, the same `i,j,k` loop order and `F32`
accumulation. Node uses `Math.fround` after each multiply and add to mirror a
single-precision accumulator; C and Bend use `float`/`F32`.

## Result

| backend | threads | positive | negative |
|---|---:|---:|---:|
| C `-O3` | 1 | 1621946231 | 4117622267 |
| Node 12 | 1 | 1621946231 | 4117622267 |
| Bend native, seq | 1 | 1621946231 | 4117622267 |
| Bend native, par4 | 1 | 1621946231 | 4117622267 |
| Bend native, par4 | 32 | 1621946231 | 4117622267 |
| Bend GPU (patched WSL2) | — | 1621946231 | 4117622267 |

No divergence: the hashes are identical. The CPU/GPU agreement is notable
because the CUDA path is compiled with `--fmad=false`; on this kernel the
operation order is simple enough that contraction does not change the result.

The earlier checksum differences (`268622200` vs `268622208` at `n=1024`) are
**not** matmul divergence: they come from summing the 1M outputs in a different
association when producing the checksum. The matmul outputs hash identically.

## Tolerance to use

- For kernel outputs: bit-exact is achievable on these cases; still compare with
  a tolerance in general, because future kernels may reassociate.
- For a single scalar checksum over 1M values: allow ~1e-7 relative; the
  association difference is ~3e-8.

Limits: `n=64`, two seeds, `F32` only. Large-magnitude values, subnormals and
non-power-of-two padding are #6's job.

Raw data: [`bench/einsum/hash.csv`](../bench/einsum/hash.csv).
