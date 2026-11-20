import Base
import ../src/mlp.bend as M
import ./real_tensor.bend as R

# Analyst pass over a real slice: 32x128 from tiny-random-gpt2
# transformer.h.0.mlp.c_fc.weight. Stats, top singular pair by power
# iteration, rank-1 edit, and the edited tensor as CSV for the judge.

type SV is Data:
  S{sigma: F32, u: List<&2, F32>, v: List<&2, F32>}

def lsq(+xs: List<&2, F32>) -> F32:
  match xs:
    case Nil{}:
      0.0
    case h <> t:
      F32.add(F32.mul(h, h), lsq(t))

def norm(+xs: List<&2, F32>) -> F32:
  F32.sqrt(lsq(xs))

def lscale(+xs: List<&2, F32>, +s: F32) -> List<&2, F32>:
  match xs:
    case Nil{}:
      Nil{}
    case h <> t:
      F32.mul(h, s) <> lscale(t, s)

def ones(n: Nat, acc: List<&2, F32>) -> List<&2, F32>:
  match n:
    case 0n:
      acc
    case 1n++p:
      ones(p, 1.0 <> acc)

def mv.go(+W: List<&2, F32>, +v: List<&2, F32>, +i: Nat, j: Nat, acc: F32) -> F32:
  match j:
    case 0n:
      acc
    case 1n++p:
      +at = Nat.add(Nat.mul(i, 128n), p)
      mv.go(W, v, i, p, F32.add(acc, F32.mul(M.lget(W, at), M.lget(v, p))))

def mv(+W: List<&2, F32>, +v: List<&2, F32>, i: Nat) -> F32:
  mv.go(W, v, i, 128n, 0.0)

def mvlist.go(i: Nat, +W: List<&2, F32>, +v: List<&2, F32>, acc: List<&2, F32>) -> List<&2, F32>:
  match i:
    case 0n:
      acc
    case 1n++p:
      mvlist.go(p, W, v, mv(W, v, p) <> acc)

def mvlist(+W: List<&2, F32>, +v: List<&2, F32>) -> List<&2, F32>:
  mvlist.go(32n, W, v, Nil{})

def mvt.go(+W: List<&2, F32>, +u: List<&2, F32>, +j: Nat, +i: Nat, acc: F32) -> F32:
  match i:
    case 0n:
      acc
    case 1n++p:
      +at = Nat.add(Nat.mul(p, 128n), j)
      mvt.go(W, u, j, p, F32.add(acc, F32.mul(M.lget(W, at), M.lget(u, p))))

def mvtlist.go(+j: Nat, +W: List<&2, F32>, +u: List<&2, F32>, acc: List<&2, F32>) -> List<&2, F32>:
  match j:
    case 0n:
      acc
    case 1n++p:
      mvtlist.go(p, W, u, mvt.go(W, u, p, 32n, 0.0) <> acc)

def mvtlist(+W: List<&2, F32>, +u: List<&2, F32>) -> List<&2, F32>:
  mvtlist.go(128n, W, u, Nil{})

def iter(n: Nat, +W: List<&2, F32>, +v: List<&2, F32>) -> SV:
  match n:
    case 0n:
      +u0 = mvlist(W, v)
      +s = norm(u0)
      S{s, lscale(u0, F32.div(1.0, s)), v}
    case 1n++p:
      +u0 = mvlist(W, v)
      +su = norm(u0)
      +u = lscale(u0, F32.div(1.0, su))
      +v0 = mvtlist(W, u)
      +sv = norm(v0)
      +v1 = lscale(v0, F32.div(1.0, sv))
      iter(p, W, v1)

def edit.go(p: Nat, +W: List<&2, F32>, +u: List<&2, F32>, +v: List<&2, F32>,
            +alpha: F32, +sigma: F32) -> List<&2, F32>:
  match p:
    case 0n:
      W
    case 1n++q:
      +i = Nat.div(q, 128n)
      +j = Nat.mod(q, 128n)
      +old = M.lget(W, q)
      +new = F32.sub(old, F32.mul(alpha, F32.mul(sigma, F32.mul(M.lget(u, i), M.lget(v, j)))))
      edit.go(q, M.lset(W, q, new), u, v, alpha, sigma)

def csv.go(+xs: List<&2, F32>, acc: String) -> String:
  match xs:
    case Nil{}:
      acc
    case h <> t:
      csv.go(t, acc ++ F32.show(h) ++ ",")

def csv(xs: List<&2, F32>) -> String:
  csv.go(xs, "")

def run(sv: SV, +W: List<&2, F32>) -> IO(Unit):
  match sv:
    case S{+sigma, u, v}:
      do IO<Unit>:
        +n : F32 = 4096.0
        +mean : F32 = F32.div(M.lsum(W), n)
        +rms : F32 = F32.sqrt(F32.div(lsq(W), n))
        +var : F32 = F32.sub(F32.mul(rms, rms), F32.mul(mean, mean))
        +std : F32 = F32.sqrt(F32.max(var, 0.0))
        IO.print("stats," ++ F32.show(mean) ++ "," ++ F32.show(std) ++ "," ++ F32.show(rms))
        IO.print("sigma," ++ F32.show(sigma))
        +W2 : List<&2, F32> = edit.go(4096n, W, u, v, 0.1, sigma)
        IO.print("edit," ++ csv(W2))

def main() -> IO(Unit):
  do IO<Unit>:
    +W : List<&2, F32> = R.tensor()
    +sv : SV = iter(6n, W, ones(128n, Nil{}))
    run(sv, W)
