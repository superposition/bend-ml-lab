import Base
import ../src/tensor.bend as T

def h(t: T.T3) -> U32:
  match t:
    case T.T3{c, a, b}:
      T.hash_bits(8, c)

def go() -> String:
  a1 : Array<F32> = T.gen(4, 7)
  b1 : Array<F32> = T.gen(8, 11)
  c1 : Array<F32> = T.matmul(a1, b1, Array.new(F32, 3n, 0.0), 2, 4, 2)
  a2 : Array<F32> = T.gen(4, 7)
  b2 : Array<F32> = T.gen(8, 11)
  t2 : T.T3 = T.matmul3(a2, b2, Array.new(F32, 3n, 0.0), 2, 4, 2)
  a3 : Array<F32> = T.gen(8, 7)
  b3 : Array<F32> = T.gen(4, 11)
  c3 : Array<F32> = T.matmul_tn(a3, b3, Array.new(F32, 3n, 0.0), 4, 2, 2)
  a4 : Array<F32> = T.gen(8, 7)
  b4 : Array<F32> = T.gen(4, 11)
  t4 : T.T3 = T.matmul3_tn(a4, b4, Array.new(F32, 3n, 0.0), 4, 2, 2)
  "mm=" ++ U32.show(T.hash_bits(8, c1)) ++ " mm3=" ++ U32.show(h(t2)) ++
  " tn=" ++ U32.show(T.hash_bits(8, c3)) ++ " tn3=" ++ U32.show(h(t4))

def main() -> IO(Unit):
  do IO<Unit>:
    IO.print(go())
