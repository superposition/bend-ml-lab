import Base
import ../../src/tensor.bend as T
import ../../src/parallel.bend as P

# Einsum family benchmarks. Prints CSV: case,checksum,ms.
# n=1024: 1.07 GFLOP per matmul. depth=8: 256 lanes.

def bench.gen(+n: U32) -> F32:
  +cells : U32 = U32.mul(n, n)
  a : Array<F32> = T.gen(cells, 7)
  b : Array<F32> = T.gen(cells, 11)
  F32.add(T.checksum(cells, a), T.checksum(cells, b))

def bench.mm(+n: U32) -> F32:
  +cells : U32 = U32.mul(n, n)
  a : Array<F32> = T.gen(cells, 7)
  b : Array<F32> = T.gen(cells, 11)
  c0 : Array<F32> = Array.new(F32, U32.log2(cells), 0.0)
  T.checksum(cells, T.matmul(a, b, c0, n, n, n))

def bench.pmm(+n: U32, depth: Nat) -> F32:
  +cells : U32 = U32.mul(n, n)
  a : Array<F32> = T.gen(cells, 7)
  b : Array<F32> = T.gen(cells, 11)
  c0 : Array<F32> = Array.new(F32, U32.log2(cells), 0.0)
  T.checksum(cells, P.pmatmul(depth, a, b, c0, n, n, n))

def bench.mmtn(+n: U32) -> F32:
  +cells : U32 = U32.mul(n, n)
  a : Array<F32> = T.gen(cells, 7)
  b : Array<F32> = T.gen(cells, 11)
  c0 : Array<F32> = Array.new(F32, U32.log2(cells), 0.0)
  T.checksum(cells, T.matmul_tn(a, b, c0, n, n, n))

def bench.outer(+n: U32) -> F32:
  +cells : U32 = U32.mul(n, n)
  a : Array<F32> = T.gen(n, 7)
  b : Array<F32> = T.gen(n, 11)
  c0 : Array<F32> = Array.new(F32, U32.log2(cells), 0.0)
  T.checksum(cells, T.outer(n, a, b, c0))

def bench.dot(+k: U32) -> F32:
  a : Array<F32> = T.gen(k, 7)
  b : Array<F32> = T.gen(k, 11)
  T.vdot(k, a, b)

def line(name: String, v: F32, t0: Nat, t1: Nat) -> IO(Unit):
  do IO<Unit>:
    IO.print(name ++ "," ++ F32.show(v) ++ "," ++ Nat.show(Nat.sub(t1, t0)))

def main() -> IO(Unit):
  do IO<Unit>:
    +n : U32 = 1024
    t0 : Nat <- IO.now()
    g : F32 = bench.gen(n)
    t1 : Nat <- IO.now()
    line("gen", g, t0, t1)

    t2 : Nat <- IO.now()
    m : F32 = bench.mm(n)
    t3 : Nat <- IO.now()
    line("matmul_seq", m, t2, t3)

    t4 : Nat <- IO.now()
    p : F32 = bench.pmm(n, 8n)
    t5 : Nat <- IO.now()
    line("matmul_par256", p, t4, t5)

    t6 : Nat <- IO.now()
    q : F32 = bench.mmtn(n)
    t7 : Nat <- IO.now()
    line("matmul_tn", q, t6, t7)

    t8 : Nat <- IO.now()
    o : F32 = bench.outer(n)
    t9 : Nat <- IO.now()
    line("outer", o, t8, t9)

    t10 : Nat <- IO.now()
    d : F32 = bench.dot(U32.mul(n, n))
    t11 : Nat <- IO.now()
    line("vdot", d, t10, t11)
