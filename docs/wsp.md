# Weight-space programming with Bend

Weight-space programming (WSP) is the idea that a model can read another
network's raw weights and write or edit them from a natural-language goal,
without gradients. It is a *concept*, not a method: hand-set weights and Tracr
compiles are WSP too. The 2026 wave added the claim that a general frontier
model can do it natively.

This repo uses Bend for the part it is actually good at: **tensor arithmetic,
statistics, and edits over weights**. It deliberately does not use Bend as the
judge.

## The rule: the serving engine is the judge

A WSP result is only comparable when the model that produced the number is the
model that will serve the edit, with the same runtime, build, sampling settings
and token cap. Different kernels produce different logits; a second engine is a
different experiment.

So the split is:

| role | tool | what it may do |
|---|---|---|
| analyst | Bend | read tensors, compute stats/deltas, propose and apply an edit, run the model's own forward pass on a fixed batch |
| judge | the serving runtime (vLLM / llama.cpp / the HF model's own engine) | score the benchmark, produce the final number |

This repo only implements the analyst. The numbers in the README are losses from
the toy model's own forward pass, not a benchmark score and not a capability
claim.

## What `wsp/analyze.bend` does

1. Trains the 2-2-8-4 MLP for 20 epochs (`src/mlp.bend`).
2. Reports `W2` statistics: mean, std, rms, min, max.
3. Applies one edit at a time to `W2`:
   - scale one class row by 1.1 (four probes);
   - add a deterministic random direction with ε = 0.05.
4. Reprices every edited model with the **same** `eval` function used for the
   baseline (mean cross-entropy over the same 256 samples).

Run:

```bash
bend wsp/analyze.bend
```

Result:

| probe | mean loss | delta |
|---|---:|---:|
| baseline | 0.002368 | — |
| row 0 × 1.1 | 0.002279 | −3.8% |
| row 1 × 1.1 | 0.002240 | −5.4% |
| row 2 × 1.1 | 0.001996 | −15.7% |
| row 3 × 1.1 | 0.002193 | −7.4% |
| random direction ε = 0.05 | 0.002451 | +3.5% |

`W2` stats: mean 0.0221, std 0.9514, rms 0.9517, min −1.5616, max 1.7637.

Interpretation: at this point in training, class row 2 is the most sensitive
row, and a same-size random perturbation is worse than any row scale. That is a
*directional sensitivity* observation on one toy checkpoint. It says nothing
about whether an LLM can find such a direction, and nothing about a real model.

## The protocol for a real checkpoint

The missing piece is I/O, not math. To run this against a real model:

1. **Export one tensor to a flat `F32` array.** For GGUF, `graft dump MODEL
   TENSOR --json` already reads Q5_K/Q6_K without dequantizing to a file; for
   safetensors, `safetensors.numpy.load_file` gives a NumPy array. Write it in a
   format the analyst can read (start with decimal text; move to raw bytes with
   a bitcast foreign once the round-trip is measured).
2. **Compute the direction/edit in Bend.** Stats, cosine, difference of means,
   power iteration for a top singular vector, rank-k truncation, row/column
   scaling. Bend's `F32` and flat arrays are sufficient; `src/tensor.bend`
   already has the kernels.
3. **Write the edited tensor back** and rebuild the checkpoint. Record the
   quantization delta: for a K-quant checkpoint, requantization changes bytes
   even for a `factor = 1.0` edit, and that error can be larger than a small WSP
   edit. Do not attribute a change to the edit before measuring the no-op
   round-trip.
4. **Judge with the serving engine.** Same engine, build, temperature, seed,
   token cap, samples per problem. For chat models, fixed prompts and a fixed
   decoding configuration; for a base model, the exact continuation protocol.
5. **Log all of it**: engine, build hash, sampling knobs, token cap, sample
   count, and the edit's byte-level diff.

## Controls that make an edit a result

Without these, a WSP number is a story:

- **Paired test.** Same problems, same decoding, edit vs no edit.
- **Size-matched random direction.** A random unit vector scaled to the same
  L2 norm as the proposed direction. If random wins, the "direction" is noise.
- **Random-trit / random-row control** for quantized checkpoints. The edit's
  support should be compared to an equally sized edit somewhere irrelevant.
- **Sealed held-out set.** The direction must not be chosen on the test set.
- **Forward-pass budget.** Count every forward pass the agent used to find the
  edit. A search that costs 10k forwards is a different result from a one-shot
  read.
- **No in-kernel scoring.** If the agent's code can call the evaluator, the
  transcript is not evidence. The `lm_understand-logs` audit trail
  (`audit.tool_free`) is the model to copy.

## Integrity traps

Three ways a WSP result can be wrong without a line of code being wrong:

1. **Permutation symmetry.** A network and its permuted twin compute the same
   function. An edit that "finds" a permuted copy has changed nothing. Compare
   outputs, not weight cosine, when the architecture admits symmetry.
2. **In-kernel cheating.** The weight-writing program has access to the eval
   harness. Audit the tools the agent can call.
3. **Tiny backdoors.** An edit can install a trigger that only the test set
   hits. Use held-out prompts and inspect the diff size and support.

## What the published WSP evidence supports

Keep the three claims separate:

| claim | evidence |
|---|---|
| models **read** weights | small networks, 17K-172K weights; replicated |
| models **write from scratch** | MinAtar/Go/car policies start at zero; replicated at 87-97% of the original score |
| models **edit** trained weights | the MATH-500 story: measurement real, attribution contested; the unedited model with thinking off already scores ~44-47%, so the edit removes a repetition-loop pathology rather than adding math ability |

Ground truth for the edit claim: `RikiyaT/lm_understand-logs` (the audit trail),
`openwsp/openwsp` (independent replication), and the eval protocol is the
authors' vendored `evaluate.py` + `math-verify`, not `lm-evaluation-harness`.
A number from a different harness is a different experiment.

## Next steps in this repo

- `mm_nt` and `transpose` to complete the backprop set (`docs/einsum.md`).
- A tensor reader for a plain `F32` file, plus a Python exporter; the current
  analyst is in-memory only.
- Power iteration for the top singular value/vector of `W2` and a rank-k
  truncation, to move from row scaling to a real subspace edit.
- A difference-of-means direction on **activations** (not weights), which is
  the cheapest published WSP-style recipe; it needs a forward pass, so it is a
  natural second kernel.
- A `--resume` checkpoint path, so the analyst can run on a model produced by a
  separate training session.
