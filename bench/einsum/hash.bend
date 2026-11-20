import Base
import ../../src/tensor.bend as T
import ../../src/parallel.bend as P

# Cross-backend bit-exactness probe: 64x64 F32 matmul, FNV-1a over the
# output bits. Run the same generator/loop order in C and Node.
def main() -> IO(Unit):
  do IO<Unit>:
    +n : U32 = 64
    +cells : U32 = U32.mul(n, n)
    a : Array<F32> = T.gen(cells, 7)
    b : Array<F32> = T.gen(cells, 11)
    c0 : Array<F32> = Array.new(F32, U32.log2(cells), 0.0)
    c : Array<F32> = T.matmul(a, b, c0, n, n, n)
    IO.print("seq," ++ U32.show(T.hash_bits(cells, c)))

    a2 : Array<F32> = T.gen(cells, 7)
    b2 : Array<F32> = T.gen(cells, 11)
    c2 : Array<F32> = Array.new(F32, U32.log2(cells), 0.0)
    d : Array<F32> = P.pmatmul(4n, a2, b2, c2, n, n, n)
    IO.print("par4," ++ U32.show(T.hash_bits(cells, d)))

    a3 : Array<F32> = T.gen(cells, 7)
    b3 : Array<F32> = T.gen(cells, 11)
    c3 : Array<F32> = Array.new(F32, U32.log2(cells), 0.0)
    e : Array<F32> = P.pmatmul!(4n, a3, b3, c3, n, n, n)
    IO.print("gpu," ++ U32.show(T.hash_bits(cells, e)))

    a4 : Array<F32> = T.gen2(cells, 13)
    b4 : Array<F32> = T.gen2(cells, 17)
    c4 : Array<F32> = Array.new(F32, U32.log2(cells), 0.0)
    f : Array<F32> = P.pmatmul!(4n, a4, b4, c4, n, n, n)
    IO.print("gpu_neg," ++ U32.show(T.hash_bits(cells, f)))
