import Base
import ../src/tensor.bend as T

def h(t: T.T3) -> U32:
  match t:
    case T.T3{c, a, b}:
      T.hash_bits(8, c)

def go() -> String:
  a1 : Array<F32> = T.gen(4, 7)
  b1 : Array<F32> = T.gen(8, 11)   # 3x2 padded 8
  bt : Array<F32> = T.transpose(3, 2, b1, Array.new(F32, 3n, 0.0))
  c1 : Array<F32> = T.matmul(a1, bt, Array.new(F32, 3n, 0.0), 2, 3, 2)
  a2 : Array<F32> = T.gen(4, 7)
  b2 : Array<F32> = T.gen(8, 11)
  t2 : T.T3 = T.matmul3_nt(a2, b2, Array.new(F32, 3n, 0.0), 2, 3, 2)
  "mm=" ++ U32.show(T.hash_bits(8, c1)) ++ " nt=" ++ U32.show(h(t2))

def main() -> IO(Unit):
  do IO<Unit>:
    IO.print(go())
