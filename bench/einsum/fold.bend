import Base
import ../../src/tensor.bend as T
import ../../src/parallel.bend as P

# 1M F32 values: indexed read vs structural tree fold vs list fold vs
# unchecked index. Each method gets a fresh array; generation is outside
# the timed region. Build with the patched WSL2 compiler for the GPU lane.
def line(name: String, v: F32, t0: Nat, t1: Nat) -> IO(Unit):
  do IO<Unit>:
    IO.print(name ++ "," ++ F32.show(v) ++ "," ++ Nat.show(Nat.sub(t1, t0)))

def main() -> IO(Unit):
  do IO<Unit>:
    +cells : U32 = 1048576

    a1 : Array<F32> = T.gen(cells, 7)
    t0 : Nat <- IO.now()
    v1 : F32 = T.checksum!(cells, a1)
    t1 : Nat <- IO.now()
    line("indexed", v1, t0, t1)

    a2 : Array<F32> = T.gen(cells, 7)
    t2 : Nat <- IO.now()
    v2 : F32 = T.sum_tree!(a2, 0.0)
    t3 : Nat <- IO.now()
    line("structural", v2, t2, t3)

    a3 : Array<F32> = T.gen(cells, 7)
    t4 : Nat <- IO.now()
    v3 : F32 = P.csumu!(cells, a3)
    t5 : Nat <- IO.now()
    line("unchecked", v3, t4, t5)

    a4 : Array<F32> = T.gen(cells, 7)
    l : List<&1, F32> = Array.to_list(~F32, a4)
    t6 : Nat <- IO.now()
    v4 : F32 = T.sum_list!(l, 0.0)
    t7 : Nat <- IO.now()
    line("list", v4, t6, t7)
