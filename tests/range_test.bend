import Base
import ../src/tensor.bend as T
import ../src/parallel.bend as P

def run(depth_note: String, c: Array<F32>) -> String:
  depth_note ++ "=" ++ T.showA(c)

def go() -> String:
  +n : U32 = 4
  +cells : U32 = 16
  a1 : Array<F32> = T.gen(cells, 7)
  b1 : Array<F32> = T.gen(cells, 11)
  bt1 : Array<F32> = T.transpose(n, n, b1, Array.new(F32, 4n, 0.0))
  c1 : Array<F32> = T.matmul_bt(a1, bt1, Array.new(F32, 4n, 0.0), n, n, n)
  a2 : Array<F32> = T.gen(cells, 7)
  b2 : Array<F32> = T.gen(cells, 11)
  bt2 : Array<F32> = T.transpose(n, n, b2, Array.new(F32, 4n, 0.0))
  c2 : Array<F32> = P.pmatmul2_bt(2n, 0n, a2, bt2, Array.new(F32, 4n, 0.0), n, n, n)
  a3 : Array<F32> = T.gen(cells, 7)
  b3 : Array<F32> = T.gen(cells, 11)
  bt3 : Array<F32> = T.transpose(n, n, b3, Array.new(F32, 4n, 0.0))
  c3 : Array<F32> = P.pmatmul2_bt(1n, 1n, a3, bt3, Array.new(F32, 4n, 0.0), n, n, n)
  a4 : Array<F32> = T.gen(cells, 7)
  b4 : Array<F32> = T.gen(cells, 11)
  bt4 : Array<F32> = T.transpose(n, n, b4, Array.new(F32, 4n, 0.0))
  c4 : Array<F32> = P.pmatmul3_bt(1n, 1n, a4, bt4, Array.new(F32, 4n, 0.0), n, n, n)
  a5 : Array<F32> = T.gen(cells, 7)
  b5 : Array<F32> = T.gen(cells, 11)
  bt5 : Array<F32> = T.transpose(n, n, b5, Array.new(F32, 4n, 0.0))
  c5 : Array<F32> = P.pmatmul4_bt(0n, 0n, a5, bt5, Array.new(F32, 4n, 0.0), n, n, n)
  run("bt", c1) ++ "|" ++ run("2d40", c2) ++ "|" ++ run("2d22", c3) ++ "|" ++ run("tile", c4) ++ "|" ++ run("tile4", c5)

def main() -> IO(Unit):
  do IO<Unit>:
    IO.print(go())
