import Base
import ./LAWS.bend as Laws
import ./src/mlp.bend as M

def Laws.lset_get(n, v):
  match n:
    case 0n:
      {==}
    case 1n+p:
      %Laws.lset_get(p, v) : {M.lget(M.lset(0.0 <> List.replicate(F32, p, 0.0), p, v), p) == _ : F32}
      {==}
