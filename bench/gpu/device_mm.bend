import Base
import ../../src/tensor.bend as T

# Device-buffer matmul: the C effect copies the two Bend Array<F32> blocks to
# device memory, runs the tiled kernel, and writes a new Array<F32>.
def Device.mm(a: Array<F32>, b: Array<F32>, c: Array<F32>, m: U32, n: U32, k: U32) -> IO(Array<F32>):
  import "./device_mm.c"
  import "./device_mm.js"

def main() -> IO(Unit):
  do IO<Unit>:
    +n : U32 = 1024
    +cells : U32 = U32.mul(n, n)
    a : Array<F32> = T.gen(cells, 7)
    b : Array<F32> = T.gen(cells, 11)
    c0 : Array<F32> = Array.new(F32, U32.log2(cells), 0.0)
    t0 : Nat <- IO.now()
    c : Array<F32> <- Device.mm(a, b, c0, n, n, n)
    t1 : Nat <- IO.now()
    v : F32 = T.checksum(cells, c)
    IO.print("cold," ++ F32.show(v) ++ "," ++ Nat.show(Nat.sub(t1, t0)))

    a2 : Array<F32> = T.gen(cells, 7)
    b2 : Array<F32> = T.gen(cells, 11)
    c2 : Array<F32> = Array.new(F32, U32.log2(cells), 0.0)
    t2 : Nat <- IO.now()
    d : Array<F32> <- Device.mm(a2, b2, c2, n, n, n)
    t3 : Nat <- IO.now()
    v2 : F32 = T.checksum(cells, d)
    IO.print("warm," ++ F32.show(v2) ++ "," ++ Nat.show(Nat.sub(t3, t2)))
