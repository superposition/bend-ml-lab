import Base
import ../../src/tensor.bend as T
import ../../src/parallel.bend as P

# Two independent 512x512 GPU matmuls: sequential vs one parallel let.
# Both variants include the two checksums in the timed region.

def seq.run(+n: U32, +cells: U32, a1: Array<F32>, bt1: Array<F32>, c1: Array<F32>,
            a2: Array<F32>, bt2: Array<F32>, c2: Array<F32>) -> F32:
  d1 : Array<F32> = P.pmatmul3_bt!(8n, 1n, a1, bt1, c1, n, n, n)
  d2 : Array<F32> = P.pmatmul3_bt!(8n, 1n, a2, bt2, c2, n, n, n)
  +s1 : F32 = T.checksum(cells, d1)
  +s2 : F32 = T.checksum(cells, d2)
  F32.add(s1, s2)

def par.run(+n: U32, +cells: U32, a1: Array<F32>, bt1: Array<F32>, c1: Array<F32>,
            a2: Array<F32>, bt2: Array<F32>, c2: Array<F32>) -> F32:
  l r = P.pmatmul3_bt!(8n, 1n, a1, bt1, c1, n, n, n)
        P.pmatmul3_bt!(8n, 1n, a2, bt2, c2, n, n, n)
  +s1 : F32 = T.checksum(cells, l)
  +s2 : F32 = T.checksum(cells, r)
  F32.add(s1, s2)

def main() -> IO(Unit):
  do IO<Unit>:
    +n : U32 = 512
    +cells : U32 = U32.mul(n, n)

    a1 : Array<F32> = T.gen(cells, 7)
    b1 : Array<F32> = T.gen(cells, 11)
    bt1 : Array<F32> = T.transpose(n, n, b1, Array.new(F32, U32.log2(cells), 0.0))
    c1 : Array<F32> = Array.new(F32, U32.log2(cells), 0.0)
    a2 : Array<F32> = T.gen(cells, 13)
    b2 : Array<F32> = T.gen(cells, 17)
    bt2 : Array<F32> = T.transpose(n, n, b2, Array.new(F32, U32.log2(cells), 0.0))
    c2 : Array<F32> = Array.new(F32, U32.log2(cells), 0.0)
    t0 : Nat <- IO.now()
    s1 : F32 = seq.run(n, cells, a1, bt1, c1, a2, bt2, c2)
    t1 : Nat <- IO.now()
    IO.print("seq," ++ Nat.show(Nat.sub(t1, t0)) ++ "," ++ F32.show(s1))

    a3 : Array<F32> = T.gen(cells, 7)
    b3 : Array<F32> = T.gen(cells, 11)
    bt3 : Array<F32> = T.transpose(n, n, b3, Array.new(F32, U32.log2(cells), 0.0))
    c3 : Array<F32> = Array.new(F32, U32.log2(cells), 0.0)
    a4 : Array<F32> = T.gen(cells, 13)
    b4 : Array<F32> = T.gen(cells, 17)
    bt4 : Array<F32> = T.transpose(n, n, b4, Array.new(F32, U32.log2(cells), 0.0))
    c4 : Array<F32> = Array.new(F32, U32.log2(cells), 0.0)
    t2 : Nat <- IO.now()
    s2 : F32 = par.run(n, cells, a3, bt3, c3, a4, bt4, c4)
    t3 : Nat <- IO.now()
    IO.print("par," ++ Nat.show(Nat.sub(t3, t2)) ++ "," ++ F32.show(s2))
