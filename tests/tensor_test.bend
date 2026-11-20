import Base
import ../src/tensor.bend as T

def go() -> String:
  +n : U32 = 4
  +cells : U32 = U32.mul(n, n)
  a1 : Array<F32> = T.gen(cells, 7)
  b1 : Array<F32> = T.gen(cells, 11)
  o : Array<F32> = T.outer(n, a1, b1, Array.new(F32, 4n, 0.0))
  a2 : Array<F32> = T.gen(cells, 7)
  rs : Array<F32> = T.rowsum(n, n, a2, Array.new(F32, 2n, 0.0))
  a3 : Array<F32> = T.gen(cells, 7)
  b3 : Array<F32> = T.gen(cells, 11)
  "outer=" ++ T.showA(o) ++ " rowsum=" ++ T.showA(rs) ++ " dot=" ++ F32.show(T.vdot(cells, a3, b3))

def main() -> IO(Unit):
  do IO<Unit>:
    IO.print(go())
