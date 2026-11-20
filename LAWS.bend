import Base
import ./src/mlp.bend as M

# LAW: setting index n of a list of length 1+n, then reading it, returns v.
law lset_get:
  for n: Nat
  for v: F32
  {M.lget(M.lset(List.replicate(F32, (1n + n : Nat), 0.0), n, v), n) == v : F32}
