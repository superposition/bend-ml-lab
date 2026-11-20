import Base
import ../src/tensor.bend as T

# mlp.bend — a 2-2-8-4 MLP trained with per-sample SGD, written in pure
# Bend. The model is a `Model` record of four `List<&2, F32>` fields. Lists are
# `Data`, so the whole model is copyable: a step reads the old weights and
# builds new lists, with no array aliasing and no ownership puzzles.
#
# The point is the epoch loop and its cost profile, not SOTA accuracy.
# Shapes are fixed: D=2 inputs, H=8 hidden, C=4 classes, 256 samples.

type Model is Data:
  M{
    w1: List<&2, F32>,
    b1: List<&2, F32>,
    w2: List<&2, F32>,
    b2: List<&2, F32>,
    loss: F32
  }

# ---------------------------------------------------------------------------
# List helpers (small lists; O(n) access is fine here)
# ---------------------------------------------------------------------------

def lget(xs: List<&2, F32>, n: Nat) -> F32:
  match xs n:
    case Nil{} _:
      0.0
    case h <> t 0n:
      h
    case h <> t 1n+p:
      lget(t, p)

def lset(xs: List<&2, F32>, n: Nat, v: F32) -> List<&2, F32>:
  match xs n:
    case Nil{} _:
      Nil{}
    case h <> t 0n:
      v <> t
    case h <> t 1n+p:
      h <> lset(t, p, v)

def lsum(+xs: List<&2, F32>) -> F32:
  match xs:
    case Nil{}:
      0.0
    case h <> t:
      F32.add(h, lsum(t))

def lscale(+xs: List<&2, F32>, +s: F32) -> List<&2, F32>:
  match xs:
    case Nil{}:
      Nil{}
    case h <> t:
      F32.mul(h, s) <> lscale(t, s)

def lmax(+xs: List<&2, F32>, cur: F32) -> F32:
  match xs:
    case Nil{}:
      cur
    case h <> t:
      lmax(t, F32.max(cur, h))

def lexp.go(+xs: List<&2, F32>, +mx: F32, acc: List<&2, F32>) -> List<&2, F32>:
  match xs:
    case Nil{}:
      acc
    case h <> t:
      lexp.go(t, mx, F32.exp(F32.sub(h, mx)) <> acc)

def ldiv.go(+xs: List<&2, F32>, +s: F32) -> List<&2, F32>:
  match xs:
    case Nil{}:
      Nil{}
    case h <> t:
      F32.div(h, s) <> ldiv.go(t, s)

def softmax(+y: List<&2, F32>) -> List<&2, F32>:
  mx = lmax(y, F32.neg(1000000000.0))
  +e : List<&2, F32> = List.reverse(&2, F32, lexp.go(y, mx, Nil{}))
  s = lsum(e)
  ldiv.go(e, s)

# p - onehot(label), the cross-entropy gradient.
def onehot.if(yes: Bool) -> F32:
  match yes:
    case True{}:
      1.0
    case False{}:
      0.0

def lgrad.go(c: Nat, +p: List<&2, F32>, +label: Nat, acc: List<&2, F32>) -> List<&2, F32>:
  match c:
    case 0n:
      acc
    case 1n++q:
      +v = F32.sub(lget(p, q), onehot.if(Nat.is_eq(q, label)))
      lgrad.go(q, p, label, v <> acc)

def lgrad(+p: List<&2, F32>, label: Nat, n: Nat) -> List<&2, F32>:
  lgrad.go(n, p, label, Nil{})

# ---------------------------------------------------------------------------
# Deterministic data
# ---------------------------------------------------------------------------

def b2u(b: Bool) -> U32:
  match b:
    case False{}:
      0
    case True{}:
      1

def sel(a: F32, b: F32, cond: Bool) -> F32:
  match cond:
    case True{}:
      a
    case False{}:
      b

# One coordinate of sample i: a cluster centre plus deterministic noise.
def coord(+i: U32, +bit: U32) -> F32:
  +label = U32.and(i, 3)
  +zero = U32.is_zero(U32.and(label, bit))
  +center = sel(0.8, F32.neg(0.8), zero)
  +s = T.prng(U32.add(U32.mul(i, 2654435761), U32.add(17, U32.mul(bit, 7))))
  +n = T.unit(s)
  F32.add(center, F32.mul(0.45, F32.sub(n, 0.5)))

def relu1(+x: F32) -> F32:
  T.relu.sel(x, F32.is_lt(x, 0.0))

# ---------------------------------------------------------------------------
# Init
# ---------------------------------------------------------------------------

def init.rev(fuel: Nat, seed: U32, acc: List<&2, F32>) -> List<&2, F32>:
  match fuel:
    case 0n:
      acc
    case 1n++p:
      +s = T.prng(seed)
      init.rev(p, s, F32.mul(T.unit2(s), 0.5) <> acc)

def mk(n: Nat, seed: U32) -> List<&2, F32>:
  List.reverse(&2, F32, init.rev(n, seed, Nil{}))

def init(seed: U32) -> Model:
  +s1 = T.prng(seed)
  +s2 = T.prng(s1)
  +s3 = T.prng(s2)
  +s4 = T.prng(s3)
  M{mk(16n, s1), mk(8n, s2), mk(32n, s3), mk(4n, s4), 0.0}

# ---------------------------------------------------------------------------
# Forward
# ---------------------------------------------------------------------------

def hval(+m: Model, +j: Nat, +x0: F32, +x1: F32) -> F32:
  match m:
    case M{w1, b1, w2, b2, loss}:
      +i = Nat.mul(j, 2n)
      +v = F32.add(
        F32.add(F32.mul(lget(w1, i), x0), F32.mul(lget(w1, (1n + i : Nat)), x1)),
        lget(b1, j)
      )
      relu1(v)

def hlist.go(j: Nat, +m: Model, +x0: F32, +x1: F32, acc: List<&2, F32>) -> List<&2, F32>:
  match j:
    case 0n:
      acc
    case 1n++p:
      hlist.go(p, m, x0, x1, hval(m, p, x0, x1) <> acc)

def hlist(+m: Model, +x0: F32, +x1: F32) -> List<&2, F32>:
  hlist.go(8n, m, x0, x1, Nil{})

def ysum(j: Nat, +m: Model, +h: List<&2, F32>, +c: Nat, acc: F32) -> F32:
  match j:
    case 0n:
      acc
    case 1n++p:
      match m:
        case M{w1, b1, w2, b2, loss}:
          +at = Nat.add(Nat.mul(c, 8n), p)
          ysum(p, m, h, c, F32.add(acc, F32.mul(lget(w2, at), lget(h, p))))

def yval(+m: Model, +h: List<&2, F32>, +c: Nat) -> F32:
  match m:
    case M{w1, b1, w2, b2, loss}:
      F32.add(ysum(8n, m, h, c, 0.0), lget(b2, c))

def ylist.go(c: Nat, +m: Model, +h: List<&2, F32>, acc: List<&2, F32>) -> List<&2, F32>:
  match c:
    case 0n:
      acc
    case 1n++p:
      ylist.go(p, m, h, yval(m, h, p) <> acc)

def ylist(+m: Model, +h: List<&2, F32>) -> List<&2, F32>:
  ylist.go(4n, m, h, Nil{})

# ---------------------------------------------------------------------------
# Backward
# ---------------------------------------------------------------------------

def dhsum(n: Nat, +m: Model, +g: List<&2, F32>, +j: Nat, acc: F32) -> F32:
  match n:
    case 0n:
      acc
    case 1n++p:
      match m:
        case M{w1, b1, w2, b2, loss}:
          +at = Nat.add(Nat.mul(p, 8n), j)
          dhsum(p, m, g, j, F32.add(acc, F32.mul(lget(g, p), lget(w2, at))))

# dh[j] = (sum_c g[c] * w2[c,j]) * relu'(h[j])
def dhval(+m: Model, +g: List<&2, F32>, +h: List<&2, F32>, +j: Nat) -> F32:
  +d = dhsum(4n, m, g, j, 0.0)
  +s = lget(h, j)
  sel(d, 0.0, F32.is_gt(s, 0.0))

def dhlist.go(j: Nat, +m: Model, +g: List<&2, F32>, +h: List<&2, F32>, acc: List<&2, F32>) -> List<&2, F32>:
  match j:
    case 0n:
      acc
    case 1n++p:
      dhlist.go(p, m, g, h, dhval(m, g, h, p) <> acc)

def dhlist(+m: Model, +g: List<&2, F32>, +h: List<&2, F32>) -> List<&2, F32>:
  dhlist.go(8n, m, g, h, Nil{})

# ---------------------------------------------------------------------------
# Updates (one field at a time; the list is threaded)
# ---------------------------------------------------------------------------

def upd.w2(n: Nat, +w2: List<&2, F32>, +g: List<&2, F32>, +h: List<&2, F32>, +lr: F32) -> List<&2, F32>:
  match n:
    case 0n:
      w2
    case 1n++p:
      +i = p
      +c = Nat.div(i, 8n)
      +j = Nat.mod(i, 8n)
      +v = F32.sub(lget(w2, i), F32.mul(F32.mul(lr, lget(g, c)), lget(h, j)))
      upd.w2(p, lset(w2, i, v), g, h, lr)

def upd.b2(n: Nat, +b2: List<&2, F32>, +g: List<&2, F32>, +lr: F32) -> List<&2, F32>:
  match n:
    case 0n:
      b2
    case 1n++p:
      +i = p
      +v = F32.sub(lget(b2, i), F32.mul(lr, lget(g, i)))
      upd.b2(p, lset(b2, i, v), g, lr)

def upd.w1(n: Nat, +w1: List<&2, F32>, +dh: List<&2, F32>, +x0: F32, +x1: F32, +lr: F32) -> List<&2, F32>:
  match n:
    case 0n:
      w1
    case 1n++p:
      +i = p
      +j = Nat.div(i, 2n)
      +d = Nat.mod(i, 2n)
      +x = sel(x0, x1, Nat.is_eq(d, 0n))
      +v = F32.sub(lget(w1, i), F32.mul(F32.mul(lr, lget(dh, j)), x))
      upd.w1(p, lset(w1, i, v), dh, x0, x1, lr)

def upd.b1(n: Nat, +b1: List<&2, F32>, +dh: List<&2, F32>, +lr: F32) -> List<&2, F32>:
  match n:
    case 0n:
      b1
    case 1n++p:
      +i = p
      +v = F32.sub(lget(b1, i), F32.mul(lr, lget(dh, i)))
      upd.b1(p, lset(b1, i, v), dh, lr)

# ---------------------------------------------------------------------------
# One sample / one epoch / evaluation
# ---------------------------------------------------------------------------

def step(+m: Model, +i: U32, +lr: F32) -> Model:
  match m:
    case M{w1, b1, w2, b2, loss}:
      +x0 : F32 = coord(i, 1)
      +x1 : F32 = coord(i, 2)
      +h : List<&2, F32> = hlist(m, x0, x1)
      +y : List<&2, F32> = ylist(m, h)
      +p : List<&2, F32> = softmax(y)
      +label : Nat = U32.to_nat(U32.and(i, 3))
      +loss2 : F32 = F32.neg(F32.log(lget(p, label)))
      +g : List<&2, F32> = lgrad(p, label, 4n)
      +dh : List<&2, F32> = dhlist(m, g, h)
      nw1 : List<&2, F32> = upd.w1(16n, w1, dh, x0, x1, lr)
      nb1 : List<&2, F32> = upd.b1(8n, b1, dh, lr)
      nw2 : List<&2, F32> = upd.w2(32n, w2, g, h, lr)
      nb2 : List<&2, F32> = upd.b2(4n, b2, g, lr)
      M{nw1, nb1, nw2, nb2, loss2}

def epoch.go(fuel: Nat, +i: U32, +m: Model, +lr: F32) -> Model:
  match fuel:
    case 0n:
      m
    case 1n++p:
      +m2 : Model = step(m, i, lr)
      epoch.go(p, U32.add(i, 1), m2, lr)

def epoch(+m: Model, +lr: F32) -> Model:
  epoch.go(256n, 0, m, lr)

def eval.go(fuel: Nat, +i: U32, +m: Model, acc: F32) -> F32:
  match fuel:
    case 0n:
      acc
    case 1n++p:
      +x0 : F32 = coord(i, 1)
      +x1 : F32 = coord(i, 2)
      +y : List<&2, F32> = ylist(m, hlist(m, x0, x1))
      +pr : List<&2, F32> = softmax(y)
      +loss : F32 = F32.neg(F32.log(lget(pr, U32.to_nat(U32.and(i, 3)))))
      eval.go(p, U32.add(i, 1), m, F32.add(acc, loss))

# Mean cross-entropy over the 256 samples.
def eval(+m: Model) -> F32:
  F32.div(eval.go(256n, 0, m, 0.0), 256.0)
