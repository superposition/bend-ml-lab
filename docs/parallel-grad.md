# Parallel batch gradients: negative

Ticket #9 asked whether mini-batch gradients can scale across lanes with
fork/join. The measurement says the work is too small for the overhead.

## Numbers

- Batched 2-layer trainer: **22 ms for 2,048 epochs** (`train/mlp_batch.csv`),
  i.e. ~0.011 ms per epoch. An epoch is 4 batches of 64 samples.
- Each batch's compute is four small matmuls: `64x2x8`, `64x8x4`, `8x64x4`,
  `2x64x8`, plus scalar loops.
- Fork/join break-even from #3: at `n=1024` (about 4 seconds of work), depth 2
  (4 lanes) gave **1.0x** and depth 4 gave 5x. At microsecond-scale work, the
  fork/join and reduction cost is the whole budget.
- Data parallelism also needs the weights read-only in every lane. Bend arrays
  are affine `Type`s; sharing them means `Array.fork` (`@unsafe`) plus a
  gradient reduction (per-lane partials or atomics). The correctness surface is
  large for a sub-0.01 ms step.

## Verdict

**Negative** for this model and batch size. The serial mini-batch is already
0.011 ms/epoch; a 4-lane split cannot recover the fork, join, and reduction
cost. The design would be worth revisiting at a batch size where one forward
pass is milliseconds (e.g., B=1024 and H=256), not 64x8.

If it is revisited: split samples, fork the weight handles read-only, accumulate
one gradient array per lane, and reduce at the join. The reduction is the part
that needs a law (associativity of the sum) and a correctness test against the
serial gradient.
