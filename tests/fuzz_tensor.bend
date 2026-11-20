import Base
import ../src/tensor.bend as T

# Differential corpus: for rectangular power-of-two shapes, compute A*B three
# ways (plain, transposed-B, transposed-A) and compare the F32-bit hashes.
# A mismatch is an index/layout bug in one of the kernels.

def case.run(+m: U32, +n: U32, +k: U32, +seed: U32) -> String:
  +cm : U32 = U32.mul(m, k)
  +cn : U32 = U32.mul(k, n)
  +cc : U32 = U32.mul(m, n)

  a1 : Array<F32> = T.gen(cm, seed)
  b1 : Array<F32> = T.gen(cn, U32.add(seed, 1))
  c1 : Array<F32> = T.matmul(a1, b1, Array.new(F32, U32.log2(cc), 0.0), m, n, k)
  +h1 : U32 = T.hash_bits(cc, c1)

  a2 : Array<F32> = T.gen(cm, seed)
  b2 : Array<F32> = T.gen(cn, U32.add(seed, 1))
  bt : Array<F32> = T.transpose(k, n, b2, Array.new(F32, U32.log2(cn), 0.0))
  c2 : Array<F32> = T.matmul_bt(a2, bt, Array.new(F32, U32.log2(cc), 0.0), m, n, k)
  +h2 : U32 = T.hash_bits(cc, c2)

  a3 : Array<F32> = T.gen(cm, seed)
  at : Array<F32> = T.transpose(m, k, a3, Array.new(F32, U32.log2(cm), 0.0))
  b3 : Array<F32> = T.gen(cn, U32.add(seed, 1))
  c3 : Array<F32> = T.matmul_tn(at, b3, Array.new(F32, U32.log2(cc), 0.0), m, n, k)
  +h3 : U32 = T.hash_bits(cc, c3)

  +ok = U32.is_eq(h1, h2) && U32.is_eq(h1, h3)
  U32.show(m) ++ "x" ++ U32.show(n) ++ "x" ++ U32.show(k) ++ "," ++
  U32.show(h1) ++ "," ++ U32.show(h2) ++ "," ++ U32.show(h3) ++ "," ++
  Bool.show(ok)

def go(fuel: Nat, +seed: U32) -> IO(Unit):
  match fuel:
    case 0n:
      IO.pure(Unit, Unit{})
    case 1n+p:
      do IO<Unit>:
        +e1 : U32 = U32.and(seed, 7)
        +e2 : U32 = U32.and(U32.shrn(seed, 2n), 7)
        +e3 : U32 = U32.and(U32.shrn(seed, 4n), 7)
        +m : U32 = U32.shln(1, U32.to_nat(e1))
        +n : U32 = U32.shln(1, U32.to_nat(e2))
        +k : U32 = U32.shln(1, U32.to_nat(e3))
        IO.print(case.run(m, n, k, seed))
        go(p, T.prng(seed))

def main() -> IO(Unit):
  do IO<Unit>:
    go(16n, 7)
