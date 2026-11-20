import Base
import ../src/tensor.bend as T
import ../train/mlp_batch.bend as L

# Every array kernel must actually transform its input. The two no-op bugs
# found during #8 (relu, map_scale) would have been caught here.

def wsum.go(fuel: Nat, ar: Array<F32> & F32, +i: U32, acc: F32) -> F32:
  match fuel:
    case 0n:
      acc
    case 1n++p:
      match ar:
        case (a1, x):
          +j = U32.add(i, 1)
          wsum.go(p, Array.get(F32, a1, j), j, F32.add(acc, F32.mul(U32.to_f32(j), x)))

def wsum(cells: U32, a: Array<F32>) -> F32:
  wsum.go(U32.to_nat(cells), Array.get(F32, a, 0), 0, 0.0)

def check(name: String, +actual: F32, +expected: F32) -> IO(Unit):
  do IO<Unit>:
    IO.print(name ++ "," ++ F32.show(actual) ++ "," ++ F32.show(expected) ++ "," ++
      Bool.show(F32.is_eq(actual, expected)))

def arr4(v0: F32, v1: F32, v2: F32, v3: F32) -> Array<F32>:
  Array.set(F32, Array.set(F32, Array.set(F32, Array.set(F32, Array.new(F32, 2n, 0.0), 0, v0), 1, v1), 2, v2), 3, v3)

def arr2(v0: F32, v1: F32) -> Array<F32>:
  Array.set(F32, Array.set(F32, Array.new(F32, 1n, 0.0), 0, v0), 1, v1)

def axpy.check(p: Array<F32> & Array<F32>) -> IO(Unit):
  match p:
    case (a, b):
      check("axpy", wsum(2, a), 10.5)

def l3.loss(l: L.L3) -> F32:
  match l:
    case L.L3{loss, y, t}:
      loss

def main() -> IO(Unit):
  do IO<Unit>:
    # relu: [1,-2,3,-4] -> [1,0,3,0]; weighted sums 1-4+9-16=-10 vs 10
    check("relu", wsum(4, T.relu(4, arr4(1.0, F32.neg(2.0), 3.0, F32.neg(4.0)))), 10.0)
    # map_scale: [1,2,3,4]*2 -> [2,4,6,8]; wsum 30 vs 60
    check("map_scale", wsum(4, T.map_scale(4, arr4(1.0, 2.0, 3.0, 4.0), 2.0)), 60.0)
    # axpy: a += 0.5*b: [1,2]+0.5*[3,4] -> [2.5,4]; wsum 10.5
    axpy.check(T.axpy(2, arr2(1.0, 2.0), arr2(3.0, 4.0), 0.5))
    # bias_add: [1,2,3,4] + [10,20] -> [11,22,13,24]; wsum 11+44+39+96=190
    check("bias_add", wsum(4, L.bias_add(4, arr4(1.0, 2.0, 3.0, 4.0), arr2(10.0, 20.0), 2)), 190.0)
    # mse3: y=[1,2], t=[0,0] -> sum 5
    check("mse", l3.loss(L.mse3(2, arr2(1.0, 2.0), arr2(0.0, 0.0))), 5.0)
    # grad3: 2*(y-t)/2 = y-t -> [1,2]; wsum 5
    check("grad", wsum(2, L.grad3(2, arr2(1.0, 2.0), arr2(0.0, 0.0))), 5.0)
    # colsum of [1,2,3,4] with C=2 -> [4,6]; wsum 16
    check("colsum", wsum(2, L.colsum(2, 2, arr4(1.0, 2.0, 3.0, 4.0))), 16.0)
    # sgd: [1,2] - 0.5*[1,1] -> [0.5,1.5]; wsum 3.5
    check("sgd", wsum(2, L.sgd(2, arr2(1.0, 2.0), arr2(1.0, 1.0), 0.5)), 3.5)
    # mask: dh=[1,2], h=[1,-1] -> [1,0]; wsum 1
    check("mask", wsum(2, L.mask(2, arr2(1.0, 2.0), arr2(1.0, F32.neg(1.0)))), 1.0)
