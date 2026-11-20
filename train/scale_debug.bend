import Base
import ../src/tensor.bend as T

def main() -> IO(Unit):
  do IO<Unit>:
    IO.print("gen," ++ T.showA(T.gen2(8, 7)))
    IO.print("scaled," ++ T.showA(T.map_scale(8, T.gen2(8, 7), 0.5)))
