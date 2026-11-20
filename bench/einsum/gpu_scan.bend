import Base
import ../../src/tensor.bend as T

# Diagnostic: one sequential pass over 1M F32 values on the GPU.
def compute() -> F32:
  +cells : U32 = 1048576
  a : Array<F32> = T.gen(cells, 7)
  T.checksum!(cells, a)

def main() -> IO(Unit):
  do IO<Unit>:
    t0 : Nat <- IO.now()
    v : F32 = compute()
    t1 : Nat <- IO.now()
    IO.print(F32.show(v) ++ "," ++ Nat.show(Nat.sub(t1, t0)))
