import Base
import ../src/mlp.bend as M
import ./checkpoint.bend as C

def train.n(n: Nat, +m: M.Model, +lr: F32) -> M.Model:
  match n:
    case 0n:
      m
    case 1n+p:
      train.n(p, M.epoch(m, lr), lr)

def main() -> IO(Unit):
  do IO<Unit>:
    +lr : F32 = 0.05
    +m0 : M.Model = C.model()
    m5 : M.Model = train.n(5n, m0, lr)
    IO.print("resume," ++ F32.show(M.eval(m0)) ++ "," ++ F32.show(M.eval(m5)))
