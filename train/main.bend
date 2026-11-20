import Base
import ../src/mlp.bend as M

# Prints epoch,mean_cross_entropy,epoch_ms as CSV. 256 samples, H=8, C=4.
def run(epoch: Nat, +total: Nat, +m: M.Model, +lr: F32) -> IO(Unit):
  match epoch:
    case 0n:
      IO.pure(Unit, Unit{})
    case 1n++p:
      do IO<Unit>:
        t0 : Nat <- IO.now()
        +m2 : M.Model = M.epoch(m, lr)
        t1 : Nat <- IO.now()
        l : F32 = M.eval(m2)
        IO.print(
          Nat.show(Nat.sub(total, p)) ++ "," ++ F32.show(l) ++ "," ++
          Nat.show(Nat.sub(t1, t0))
        )
        run(p, total, m2, lr)

def main() -> IO(Unit):
  do IO<Unit>:
    run(20n, 20n, M.init(42), 0.05)
