# Training epochs in Bend

`src/mlp.bend` is a 2-2-8-4 MLP trained with per-sample SGD, written as pure
Bend over `List<&2, F32>` weights. It exists to answer one question: what does an
epoch cost, and does the cost stay linear when you run a session of many epochs?

It is not a fast trainer. The matmul track is the fast path; this is the
reference path where the update rule is legible.

## The model

Shapes are fixed and all powers of two:

- `D = 2` inputs, `H = 8` hidden units, `C = 4` classes, `N = 256` samples.
- `W1: H x D` (16 values), `b1: H` (8), `W2: C x H` (32), `b2: C` (4).
- The model is a `Data` record of four `List<&2, F32>` fields and a `loss`
  field. Being `Data`, it is copyable; every update builds new lists.

Data is deterministic and generated inside the program:

- sample `i` has label `i mod 4`;
- cluster centres are the corners `(±0.8, ±0.8)`, selected by the label bits;
- each coordinate adds `0.45 * (u - 0.5)` where `u` is a xorshift32 draw;
- 64 samples per class, 256 per epoch.

## Forward and backward

For one sample `(x, label)`:

```
h_j = relu( sum_d W1[j,d] x_d + b1[j] )              j = 0..7
y_c = sum_j W2[c,j] h_j + b2[c]                       c = 0..3
p   = softmax(y)
L   = -log p[label]

g_c     = p_c - 1[c = label]                          (cross-entropy gradient)
db2_c   = g_c
dW2[c,j]= g_c h_j
dh_j    = ( sum_c g_c W2[c,j] ) * 1[h_j > 0]
db1_j   = dh_j
dW1[j,d]= dh_j x_d

W -= lr * dW,   b -= lr * db
```

Every operation is a recursive walk with a `Nat` fuel. `List.get`/`List.set`
are O(n); at 60 weights that is irrelevant. The update functions take the fuel
first so the termination checker accepts the rewritten list (`docs/bend-notes.md`
§3).

Run:

```bash
bend train/main.bend            # JS, ~12 ms for the first epoch
bend train/main.bend -o /tmp/train && /tmp/train
```

Output is CSV: `epoch, mean_cross_entropy, epoch_ms`.

## Epoch cost

`train/epoch_sweep.sh`, native build, 32 vCPU WSL2:

| epochs | final loss | total ms | ms/epoch |
|---:|---:|---:|---:|
| 1 | 0.2537 | 6 | 6.0 |
| 2 | 0.0883 | 9 | 4.5 |
| 4 | 0.0293 | 16 | 4.0 |
| 8 | 0.00962 | 30 | 3.8 |
| 16 | 0.00329 | 57 | 3.6 |
| 32 | 0.00122 | 114 | 3.6 |
| 64 | 0.000487 | 222 | 3.5 |

The first epoch pays allocation and cache warm-up; after that the marginal cost
is flat at ~3.5 ms/epoch. That is the property that matters for a session: a
100-epoch run is ~0.35 s, a 10k-epoch run ~35 s, with no superlinear term from
the list representation.

The loss curve is real (it is the same forward pass used by `eval`), but the
task is trivially separable. Do not read the absolute loss as a benchmark
quality number.

## Why lists, not arrays

`Array<F32>` is a `Type`: one owner, reads return the array, writes rebind it.
A weight update would have to thread four arrays through every helper and clone
the weights that are needed both for the forward and the update. Lists are
`Data`: copy them, rebuild the spine, pass them anywhere. For 60 weights the
asymptotics lose to the ergonomics.

The crossover is exactly what the matmul track measures. At 1024x1024, arrays
win by orders of magnitude because a list of a million `F32` is a million
pointers and a million `O(n)` walks. The right future trainer is batched: use
`src/tensor.bend` for `W1`, `W2`, and the batch activations, and keep the
epoch/session logic here.

## Checkpoint and resume

Implemented as a **decimal Bend source file** (`train/save.bend` +
`train/resume.bend`):

1. Train 5 epochs, serialize the `Model` to `train/checkpoint.bend` with
   `F32.show` (negative values wrapped in `F32.neg`), write it through
   `Process.run("sh", ["-c", "cat > ..."])`.
2. A second process imports the generated module and trains 5 more epochs.
3. Compare with the 10-epoch in-process reference.

Result: `reference 0.0067563583`, `resume 0.0067563583` — no drift at the
printed precision (`train/checkpoint.csv`, `train/checkpoint.sh`).

Caveat: decimal text is not a guarantee of bit-exactness. It round-trips this
trajectory exactly, but a bit-exact checkpoint needs `F32.from_bits` (absent
from Base) or a foreign bitcast. The checkpoint is also executable source, so
the compiler parses it; that is fine for a lab, not a general format.

The stable-RNG property helps: noise is derived from the sample index, so
resume needs only the weights, not an RNG state.

## Next steps

- Batched trainer on `src/tensor.bend`: `x @ W1`, `h @ W2`, then `dy * W2^T`
  (add `mm_nt`) and `h^T @ dy` (already `mm_tn`). This is the 1000x path.
- Mini-batch gradient accumulation with a parallel reduction over samples.
- `LAWS.bend` for the update invariants that are actually provable: index
  bounds, shape preservation, `lset`/`lget` round-trip, `softmax` sums to 1 for
  a non-empty list.
- Decimal checkpoint + `--resume`, with a round-trip test that asserts the
  reloaded model reproduces the same loss to a stated tolerance.
