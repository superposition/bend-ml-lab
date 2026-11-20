import Base
import ../src/mlp.bend as M

# Save a decimal checkpoint as a Bend source file after 5 epochs, then
# continue to 10 in-process as the reference.

def proc.out(r: U32 & (String & String)) -> String:
  match r:
    case (status, io):
      match io:
        case (o, e):
          o

def show.f32.sel(+x: F32, neg: Bool) -> String:
  match neg:
    case True{}:
      "F32.neg(" ++ F32.show(F32.neg(x)) ++ ")"
    case False{}:
      F32.show(x)

def show.f32(+x: F32) -> String:
  show.f32.sel(x, F32.is_lt(x, 0.0))

def show.list.go(+xs: List<&2, F32>, acc: String) -> String:
  match xs:
    case Nil{}:
      acc ++ "]"
    case h <> t:
      show.list.go(t, acc ++ show.f32(h) ++ ",")

def show.list(xs: List<&2, F32>) -> String:
  show.list.go(xs, "[")

def show.model(m: M.Model) -> String:
  match m:
    case M.M{w1, b1, w2, b2, loss}:
      "import Base\nimport ../src/mlp.bend as M\n\ndef model() -> M.Model:\n  M.M{" ++
      show.list(w1) ++ "," ++ show.list(b1) ++ "," ++ show.list(w2) ++ "," ++
      show.list(b2) ++ "," ++ F32.show(loss) ++ "}\n"

def write(+path: String, s: String) -> IO(Unit):
  do IO<Unit>:
    r : U32 & (String & String) <- IO.try(
      U32 & (String & String),
      Process.run("sh", ["-c", "cat > " ++ path], s, 1024, 10000)
    )
    o : String = proc.out(r)
    IO.pure(Unit, Unit{})

def train.n(n: Nat, +m: M.Model, +lr: F32) -> M.Model:
  match n:
    case 0n:
      m
    case 1n+p:
      train.n(p, M.epoch(m, lr), lr)

def main() -> IO(Unit):
  do IO<Unit>:
    +lr : F32 = 0.05
    m0 : M.Model = M.init(42)
    +m5 : M.Model = train.n(5n, m0, lr)
    src : String = show.model(m5)
    write("train/checkpoint.bend", src)
    m10 : M.Model = train.n(5n, m5, lr)
    IO.print("save," ++ F32.show(M.eval(m5)) ++ "," ++ F32.show(M.eval(m10)))
