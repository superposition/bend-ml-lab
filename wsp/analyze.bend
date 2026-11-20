import Base
import ../src/mlp.bend as M
import ../src/tensor.bend as T

# Weight-space diagnostics on the toy MLP from src/mlp.bend.
#
# Everything here is an *analyst* operation: read tensors, compute
# statistics, apply an edit, and observe the model's own forward pass.
# No claim about capability is made; see docs/wsp.md.

type Stats is Data:
  S{mean: F32, std: F32, rms: F32, lo: F32, hi: F32}

def lmean(+xs: List<&2, F32>, n: Nat) -> F32:
  F32.div(M.lsum(xs), F32.from_nat(n))

def l2(+xs: List<&2, F32>) -> F32:
  match xs:
    case Nil{}:
      0.0
    case h <> t:
      F32.add(F32.mul(h, h), l2(t))

def lmin(+xs: List<&2, F32>, cur: F32) -> F32:
  match xs:
    case Nil{}:
      cur
    case h <> t:
      lmin(t, F32.min(cur, h))

def lmax(+xs: List<&2, F32>, cur: F32) -> F32:
  match xs:
    case Nil{}:
      cur
    case h <> t:
      lmax(t, F32.max(cur, h))

def stats(+xs: List<&2, F32>, +n: Nat) -> Stats:
  +mu = lmean(xs, n)
  +sq = l2(xs)
  +var = F32.sub(F32.div(sq, F32.from_nat(n)), F32.mul(mu, mu))
  S{
    mu,
    F32.sqrt(F32.max(var, 0.0)),
    F32.sqrt(F32.div(sq, F32.from_nat(n))),
    lmin(xs, 1000000000.0),
    lmax(xs, F32.neg(1000000000.0))
  }

def cosine(+a: List<&2, F32>, +b: List<&2, F32>) -> F32:
  match a b:
    case Nil{} _:
      0.0
    case _ Nil{}:
      0.0
    case ha <> ta hb <> tb:
      F32.add(F32.mul(ha, hb), cosine(ta, tb))

def set.w2(+m: M.Model, +w2n: List<&2, F32>) -> M.Model:
  match m:
    case M.M{w1, b1, w2, b2, loss}:
      M.M{w1, b1, w2n, b2, loss}

def w2of(+m: M.Model) -> List<&2, F32>:
  match m:
    case M.M{w1, b1, w2, b2, loss}:
      w2

# Scale one output row c of W2 (a class's 8 outgoing weights) by s.
def scale.row(n: Nat, +w2: List<&2, F32>, +c: Nat, +s: F32) -> List<&2, F32>:
  match n:
    case 0n:
      w2
    case 1n++p:
      +at = Nat.add(Nat.mul(c, 8n), p)
      +v = F32.mul(M.lget(w2, at), s)
      scale.row(p, M.lset(w2, at, v), c, s)

# Add eps * d[i] to every W2 entry, where d is a deterministic direction.
def rand.dir(n: Nat, seed: U32, acc: List<&2, F32>) -> List<&2, F32>:
  match n:
    case 0n:
      acc
    case 1n+p:
      +s = T.prng(seed)
      rand.dir(p, s, T.unit2(s) <> acc)

def add.dir(n: Nat, +w2: List<&2, F32>, +d: List<&2, F32>, +eps: F32) -> List<&2, F32>:
  match n:
    case 0n:
      w2
    case 1n++p:
      +i = p
      +v = F32.add(M.lget(w2, i), F32.mul(eps, M.lget(d, i)))
      add.dir(p, M.lset(w2, i, v), d, eps)

def show.stats(name: String, s: Stats) -> String:
  match s:
    case S{mean, std, rms, lo, hi}:
      name ++ "," ++ F32.show(mean) ++ "," ++ F32.show(std) ++ "," ++
      F32.show(rms) ++ "," ++ F32.show(lo) ++ "," ++ F32.show(hi)

def summarize(name: String, +m: M.Model) -> String:
  show.stats(name, stats(w2of(m), 32n))

def train.epochs(n: Nat, +m: M.Model, +lr: F32) -> M.Model:
  match n:
    case 0n:
      m
    case 1n+p:
      train.epochs(p, M.epoch(m, lr), lr)

def main() -> IO(Unit):
  do IO<Unit>:
    m0 : M.Model = M.init(42)
    +m : M.Model = train.epochs(20n, m0, 0.05)
    base : F32 = M.eval(m)
    IO.print("metric,value")
    IO.print(summarize("w2_stats", m))
    IO.print("baseline_loss," ++ F32.show(base))
    row0 : M.Model = set.w2(m, scale.row(8n, w2of(m), 0n, 1.1))
    IO.print("scale_row0_x1.1," ++ F32.show(M.eval(row0)))
    row1 : M.Model = set.w2(m, scale.row(8n, w2of(m), 1n, 1.1))
    IO.print("scale_row1_x1.1," ++ F32.show(M.eval(row1)))
    row2 : M.Model = set.w2(m, scale.row(8n, w2of(m), 2n, 1.1))
    IO.print("scale_row2_x1.1," ++ F32.show(M.eval(row2)))
    row3 : M.Model = set.w2(m, scale.row(8n, w2of(m), 3n, 1.1))
    IO.print("scale_row3_x1.1," ++ F32.show(M.eval(row3)))
    d : List<&2, F32> = rand.dir(32n, 7, Nil{})
    rnd : M.Model = set.w2(m, add.dir(32n, w2of(m), d, 0.05))
    IO.print("random_dir_eps0.05," ++ F32.show(M.eval(rnd)))
