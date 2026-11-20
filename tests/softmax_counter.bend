import Base
import ../src/mlp.bend as M

def check(name: String, l: List<&2, F32>) -> String:
  +p : List<&2, F32> = M.softmax(l)
  +s : F32 = M.lsum(p)
  name ++ "," ++ F32.show(s) ++ "," ++ Bool.show(F32.is_eq(s, 1.0))

def main() -> IO(Unit):
  do IO<Unit>:
    IO.print(check("a", 0.1 <> 0.2 <> 0.3 <> Nil{}))
    IO.print(check("b", 1.0 <> 2.0 <> 4.0 <> Nil{}))
    IO.print(check("c", 10.0 <> 20.0 <> 30.0 <> Nil{}))
    IO.print(check("d", 0.001 <> 1000.0 <> 2000.0 <> Nil{}))
    IO.print(check("e", F32.neg(1.0) <> 0.0 <> 1.0 <> Nil{}))
    IO.print(check("f", 1.0 <> 1.0 <> 1.0 <> Nil{}))
    IO.print(check("g", 0.5 <> 0.25 <> 0.125 <> 0.0625 <> Nil{}))
