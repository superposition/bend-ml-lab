import Base
import ../../src/tensor.bend as T
import ../../src/parallel.bend as P

# B is transposed on the CPU once, so both inner-loop operands advance by 1.
# Build with the patched WSL2 compiler; run with `+RTS --gpu 4GB`.
def compute(depth: Nat) -> F32:
  +n : U32 = 1024
  +cells : U32 = U32.mul(n, n)
  a : Array<F32> = T.gen(cells, 7)
  b : Array<F32> = T.gen(cells, 11)
  bt : Array<F32> = T.transpose(n, n, b, Array.new(F32, U32.log2(cells), 0.0))
  c0 : Array<F32> = Array.new(F32, U32.log2(cells), 0.0)
  T.checksum(cells, P.pmatmul_bt!(depth, a, bt, c0, n, n, n))

def main() -> IO(Unit):
  do IO<Unit>:
    t0 : Nat <- IO.now()
    v : F32 = compute(8n)
    t1 : Nat <- IO.now()
    IO.print(F32.show(v) ++ "," ++ Nat.show(Nat.sub(t1, t0)))
