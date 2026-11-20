import Base
import ../../src/tensor.bend as T
import ../../src/parallel.bend as P

# 2D block tree: rd row levels + cd column levels = 2^(rd+cd) lanes.
# Prep (generate + transpose) is outside the timed region; the printed
# number is the GPU matmul alone. Build with the patched WSL2 compiler.
def main() -> IO(Unit):
  do IO<Unit>:
    +n : U32 = 1024
    +cells : U32 = U32.mul(n, n)
    t_prep0 : Nat <- IO.now()
    a : Array<F32> = T.gen(cells, 7)
    b : Array<F32> = T.gen(cells, 11)
    bt : Array<F32> = T.transpose(n, n, b, Array.new(F32, U32.log2(cells), 0.0))
    c0 : Array<F32> = Array.new(F32, U32.log2(cells), 0.0)
    t_prep1 : Nat <- IO.now()
    t0 : Nat <- IO.now()
    c : Array<F32> = P.pmatmul2_bt!(10n, 2n, a, bt, c0, n, n, n)
    t1 : Nat <- IO.now()
    v : F32 = T.checksum(cells, c)
    IO.print(
      F32.show(v) ++ "," ++ Nat.show(Nat.sub(t1, t0)) ++ "," ++
      Nat.show(Nat.sub(t_prep1, t_prep0))
    )
