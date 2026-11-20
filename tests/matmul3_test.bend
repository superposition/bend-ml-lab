import Base
import ../src/tensor.bend as T

def h3(t: T.T3) -> U32:
  match t:
    case T.T3{c, a, b}:
      T.hash_bits(16, c)

def go() -> String:
  +n : U32 = 4
  +cells : U32 = 16
  a : Array<F32> = T.gen(cells, 7)
  b : Array<F32> = T.gen(cells, 11)
  c1 : Array<F32> = T.matmul(a, b, Array.new(F32, 4n, 0.0), n, n, n)
  a2 : Array<F32> = T.gen(cells, 7)
  b2 : Array<F32> = T.gen(cells, 11)
  t3 : T.T3 = T.matmul3(a2, b2, Array.new(F32, 4n, 0.0), n, n, n)
  a3 : Array<F32> = T.gen(cells, 7)
  b3 : Array<F32> = T.gen(cells, 11)
  c3 : Array<F32> = T.matmul_tn(a3, b3, Array.new(F32, 4n, 0.0), n, n, n)
  a4 : Array<F32> = T.gen(cells, 7)
  b4 : Array<F32> = T.gen(cells, 11)
  t4 : T.T3 = T.matmul3_tn(a4, b4, Array.new(F32, 4n, 0.0), n, n, n)
  "mm=" ++ U32.show(T.hash_bits(cells, c1)) ++ " mm3=" ++ U32.show(h3(t3)) ++
  " tn=" ++ U32.show(T.hash_bits(cells, c3)) ++ " tn3=" ++ U32.show(h3(t4))

def main() -> IO(Unit):
  do IO<Unit>:
    IO.print(go())
