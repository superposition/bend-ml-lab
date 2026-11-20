import Base
import ../src/mlp.bend as M

def main() -> IO(Unit):
  do IO<Unit>:
    +m0 : M.Model = M.init(42)
    l0 : F32 = M.eval(m0)
    m1 : M.Model = M.epoch(m0, 0.05)
    l1 : F32 = M.eval(m1)
    IO.print(F32.show(l0) ++ " -> " ++ F32.show(l1))
