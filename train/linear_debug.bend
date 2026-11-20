import Base
import ../src/tensor.bend as T
import ./linear_batch.bend as L

def main() -> IO(Unit):
  do IO<Unit>:
    wt : Array<F32> = T.map_scale(8, T.gen2(8, 7), 0.5)
    b : Array<F32> = Array.new(F32, 2n, 0.0)
    IO.print("wt," ++ T.showA(wt))
    x : Array<F32> = L.mkx(64, 0)
    t : Array<F32> = L.mkt(64, 0)
    IO.print("x," ++ T.showA(x))
    IO.print("t0," ++ T.showA(t))

