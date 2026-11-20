# Proofs for the kernel primitives

Tickets #4 and #5. The lab now has a law, a proof, the proven kernel's verdict,
and a mutation test that shows the gate catches a planted bug.

## The law

`LAWS.bend`:

```python
law lset_get:
  for n: Nat
  for v: F32
  {M.lget(M.lset(List.replicate(F32, (1n + n : Nat), 0.0), n, v), n) == v : F32}
```

This is the round-trip used by every weight update: set a list element and read
it back. `List.replicate` gives a list of length `1+n`, so the index is always
in range; no precondition is needed.

## The proof

`PROOF.bend` is induction on `n`:

- `0n`: `lset` and `lget` both reduce at index 0; `{==}`.
- `1n+p`: unfold `replicate` to `0.0 <> replicate(p)`; the goal becomes the
  induction hypothesis at `p`. One rewrite, then `{==}`:

```python
%Laws.lset_get(p, v) : {M.lget(M.lset(0.0 <> List.replicate(F32, p, 0.0), p, v), p) == _ : F32}
{==}
```

## Checker cost

| run | time |
|---|---:|
| `bend PROOF.bend` (default checker) | 0.25 s |
| `bend PROOF.bend --verdict`, cold (Lean kernel build) | 39.81 s |
| `bend PROOF.bend --verdict`, warm | 0.38 s |

The cold cost is the one-time BendTT kernel build with Lean 4.34.0. After that
the verdict is sub-second, which is the property the project claims: an agent
can afford to run it after an edit.

## Mutation test

Changing the base case of `lset` from `v <> t` to `Nil{}` makes the proof fail:

```
SOME PROOFS FAIL
expected : 0.0
observed : v
Location: Laws.lset_get
case 0n:
  {==}
```

Restoring the definition returns `ALL PROOFS CHECK`. The proof is sensitive to
the kernel's actual behavior, not just to the types.

## Limits

The law is about the list primitive, not a full tensor shape algebra. A
matmul/transpose law would need shapes in the types; this library carries
`m,n,k` as runtime `U32`s. That is a language-level step (shape-indexed
tensors), not a missing tactic. Recorded as a negative/limit here rather than
claimed.

Raw timing: [`bench/einsum/proofs.csv`](../bench/einsum/proofs.csv).
