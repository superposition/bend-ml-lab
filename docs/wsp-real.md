# WSP on a real checkpoint (analyst + judge)

Ticket #10. A 32x128 slice of a real checkpoint goes through the full loop:

1. **Export** — `wsp/export_real.py` takes
   `hf-internal-testing/tiny-random-gpt2` ->
   `transformer.h.0.mlp.c_fc.weight[0:32,0:128]` and writes
   `wsp/real_tensor.bend` (4096 values).
2. **Analyse/edit in Bend** — `wsp/real.bend`: mean/std/rms, top singular pair
   by 6 power iterations, rank-1 edit `W - 0.1*sigma*u*v^T`, edited tensor as
   CSV.
3. **Judge** — `wsp/judge_real.py` applies the edited slice to the HF model
   and scores fixed prompts with the **same HF runtime** that serves the
   checkpoint. It also applies a random direction of the same Frobenius norm.

## Result

Bend: mean `-1.904e-4`, std `1.975e-2`, rms `1.975e-2`, sigma `0.3232`.
Edit: Frobenius norm `3.232e-2`, 2.56% of the slice.

| prompt | rank-1 max logit Δ | top-1 changed | top-5 overlap | random max Δ | random top-1 |
|---|---:|---|---:|---:|---|
| The capital of France is | 3.33e-3 | no | 5/5 | 3.80e-3 | no |
| Once upon a time | 4.50e-3 | no | 5/5 | 3.23e-3 | no |
| 2 + 2 = | 1.08e-2 | yes | 5/5 | 5.84e-3 | no |

## Reading

- The pipeline works end to end on real checkpoint bytes: Bend read, edited,
  and emitted the tensor; the model's own runtime produced the score.
- **The rank-1 direction is not privileged.** A random direction of the same
  norm produces comparable logit drift on two prompts and less on the third.
  The one top-1 flip is on a random-weight model and is not a capability
  result. The control is the result: without it, the top-1 flip would look like
  a finding.
- The judge is the same HF engine. A second serving engine is #12 and is
  blocked on this machine (no vLLM/llama.cpp path for this checkpoint).

## Reproduce

```bash
bend wsp/real.bend -o /tmp/real
/home/superposition/yue2/.venv/bin/python wsp/judge_real.py
```

Raw: [`wsp/results.csv`](../wsp/results.csv). Limits: one random-weight
checkpoint, one slice, 6 power iterations, CPU HF runtime.
