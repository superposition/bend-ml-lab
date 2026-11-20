import Base
import ../src/tensor.bend as T
import ../src/mlp.bend as M

# Batched 2-8-4 MLP: y = relu(x W1 + b1) W2 + b2, MSE, 64-sample mini-batches.
# The forward and all three weight gradients are matmuls; the chain passes
# records so no computed tuple is destructured.

type L3 is Type:
  L3{loss: F32, y: Array<F32>, t: Array<F32>}

# --- array kernels ---------------------------------------------------------

def bias_add.go(fuel: Nat, ar: Array<F32> & F32, br: Array<F32> & F32,
                +cols: U32, +i: U32) -> Array<F32>:
  match fuel:
    case 0n:
      match ar:
        case (a1, x):
          a1
    case 1n++p:
      match ar:
        case (a1, x):
          match br:
            case (b1, y):
              +c1 = U32.and(U32.add(i, 1), U32.sub(cols, 1))
              a2 = Array.set(F32, a1, i, F32.add(x, y))
              bias_add.go(p, Array.get(F32, a2, U32.add(i, 1)), Array.get(F32, b1, c1), cols, U32.add(i, 1))

def bias_add(cells: U32, a: Array<F32>, b: Array<F32>, +cols: U32) -> Array<F32>:
  bias_add.go(U32.to_nat(cells), Array.get(F32, a, 0), Array.get(F32, b, 0), cols, 0)

def mse.go(fuel: Nat, yr: Array<F32> & F32, tr: Array<F32> & F32, +i: U32, acc: F32) -> L3:
  match fuel:
    case 0n:
      match yr:
        case (y1, x):
          match tr:
            case (t1, t):
              L3{acc, y1, t1}
    case 1n++p:
      match yr:
        case (y1, x):
          match tr:
            case (t1, t):
              +d = F32.sub(x, t)
              +j = U32.add(i, 1)
              mse.go(p, Array.get(F32, y1, j), Array.get(F32, t1, j), j, F32.add(acc, F32.mul(d, d)))

def mse3(cells: U32, y: Array<F32>, t: Array<F32>) -> L3:
  mse.go(U32.to_nat(cells), Array.get(F32, y, 0), Array.get(F32, t, 0), 0, 0.0)

def grad.go(fuel: Nat, yr: Array<F32> & F32, tr: Array<F32> & F32,
            out: Array<F32>, +scale: F32, +i: U32) -> Array<F32>:
  match fuel:
    case 0n:
      out
    case 1n++p:
      match yr:
        case (y1, x):
          match tr:
            case (t1, t):
              +j = U32.add(i, 1)
              o2 = Array.set(F32, out, i, F32.mul(F32.sub(x, t), scale))
              grad.go(p, Array.get(F32, y1, j), Array.get(F32, t1, j), o2, scale, j)

def grad3(+cells: U32, y: Array<F32>, t: Array<F32>) -> Array<F32>:
  grad.go(U32.to_nat(cells), Array.get(F32, y, 0), Array.get(F32, t, 0),
    Array.new(F32, U32.log2(cells), 0.0), F32.div(2.0, U32.to_f32(cells)), 0)

def colsum.go(fuel: Nat, gr: Array<F32> & F32, orr: Array<F32> & F32,
              +cols: U32, +p: U32) -> Array<F32>:
  match fuel:
    case 0n:
      match orr:
        case (o1, y):
          o1
    case 1n++q:
      match gr:
        case (g1, x):
          match orr:
            case (o1, y):
              +c = U32.and(p, U32.sub(cols, 1))
              o2 = Array.set(F32, o1, c, F32.add(y, x))
              +np = U32.add(p, 1)
              colsum.go(q, Array.get(F32, g1, np), Array.get(F32, o2, U32.and(np, U32.sub(cols, 1))), cols, np)

def colsum(batch: U32, +cols: U32, g: Array<F32>) -> Array<F32>:
  colsum.go(U32.to_nat(U32.mul(batch, cols)), Array.get(F32, g, 0),
    Array.get(F32, Array.new(F32, U32.log2(cols), 0.0), 0), cols, 0)

def sgd.go(fuel: Nat, wr: Array<F32> & F32, gr: Array<F32> & F32,
           +lr: F32, +i: U32) -> Array<F32>:
  match fuel:
    case 0n:
      match wr:
        case (w1, x):
          w1
    case 1n++q:
      match wr:
        case (w1, x):
          match gr:
            case (g1, y):
              +j = U32.add(i, 1)
              v = F32.sub(x, F32.mul(lr, y))
              sgd.go(q, Array.get(F32, Array.set(F32, w1, i, v), j), Array.get(F32, g1, j), lr, j)

def sgd(cells: U32, w: Array<F32>, g: Array<F32>, +lr: F32) -> Array<F32>:
  sgd.go(U32.to_nat(cells), Array.get(F32, w, 0), Array.get(F32, g, 0), lr, 0)

def mask.go(fuel: Nat, dr: Array<F32> & F32, hr: Array<F32> & F32, +i: U32) -> Array<F32>:
  match fuel:
    case 0n:
      match dr:
        case (d1, x):
          d1
    case 1n++p:
      match dr:
        case (d1, x):
          match hr:
            case (h1, y):
              +j = U32.add(i, 1)
              +v = M.sel(x, 0.0, F32.is_gt(y, 0.0))
              d2 = Array.set(F32, d1, i, v)
              mask.go(p, Array.get(F32, d2, j), Array.get(F32, h1, j), j)

def mask(cells: U32, dh: Array<F32>, h: Array<F32>) -> Array<F32>:
  mask.go(U32.to_nat(cells), Array.get(F32, dh, 0), Array.get(F32, h, 0), 0)

# --- data ------------------------------------------------------------------

def fill.x(j: Nat, +off: U32, a: Array<F32>) -> Array<F32>:
  match j:
    case 0n:
      a
    case 1n++p:
      +i = U32.add(off, U32.from_nat(p))
      +at = U32.mul(U32.from_nat(p), 2)
      a1 = Array.set(F32, Array.set(F32, a, at, M.coord(i, 1)), U32.add(at, 1), M.coord(i, 2))
      fill.x(p, off, a1)

def fill.t(j: Nat, +off: U32, a: Array<F32>) -> Array<F32>:
  match j:
    case 0n:
      a
    case 1n++p:
      +i = U32.add(off, U32.from_nat(p))
      +label = U32.to_nat(U32.and(i, 3))
      +base = U32.mul(U32.from_nat(p), 4)
      a1 = Array.set(F32, a, base, M.sel(1.0, 0.0, Nat.is_eq(0n, label)))
      a2 = Array.set(F32, a1, U32.add(base, 1), M.sel(1.0, 0.0, Nat.is_eq(1n, label)))
      a3 = Array.set(F32, a2, U32.add(base, 2), M.sel(1.0, 0.0, Nat.is_eq(2n, label)))
      a4 = Array.set(F32, a3, U32.add(base, 3), M.sel(1.0, 0.0, Nat.is_eq(3n, label)))
      fill.t(p, off, a4)

def mkx(+batch: U32, off: U32) -> Array<F32>:
  fill.x(U32.to_nat(batch), off, Array.new(F32, U32.log2(U32.mul(batch, 2)), 0.0))

def mkt(+batch: U32, off: U32) -> Array<F32>:
  fill.t(U32.to_nat(batch), off, Array.new(F32, U32.log2(U32.mul(batch, 4)), 0.0))

# --- records ---------------------------------------------------------------

type F1 is Type:
  F1{h: Array<F32>, x: Array<F32>, w1: Array<F32>, b1: Array<F32>}

type F2 is Type:
  F2{y: Array<F32>, h: Array<F32>, w2: Array<F32>, x: Array<F32>, w1: Array<F32>, b1: Array<F32>, b2: Array<F32>}

type LB is Type:
  LB{loss: F32, y: Array<F32>, t: Array<F32>, h: Array<F32>, w2: Array<F32>, x: Array<F32>, w1: Array<F32>, b1: Array<F32>, b2: Array<F32>}

type GB is Type:
  GB{dy: Array<F32>, h: Array<F32>, w2: Array<F32>, x: Array<F32>, w1: Array<F32>, b1: Array<F32>, b2: Array<F32>, loss: F32}

type A is Type:
  A{dw2: Array<F32>, h: Array<F32>, dy: Array<F32>, w2: Array<F32>, x: Array<F32>, w1: Array<F32>, b1: Array<F32>, b2: Array<F32>, loss: F32}

type B is Type:
  B{dh0: Array<F32>, h: Array<F32>, dy: Array<F32>, w2: Array<F32>, x: Array<F32>, w1: Array<F32>, b1: Array<F32>, b2: Array<F32>, loss: F32, dw2: Array<F32>}

type C is Type:
  C{db2: Array<F32>, dh0: Array<F32>, h: Array<F32>, w2: Array<F32>, x: Array<F32>, w1: Array<F32>, b1: Array<F32>, b2: Array<F32>, loss: F32, dw2: Array<F32>}

type D is Type:
  D{dh: Array<F32>, w2: Array<F32>, x: Array<F32>, w1: Array<F32>, b1: Array<F32>, b2: Array<F32>, loss: F32, dw2: Array<F32>, db2: Array<F32>}

type E is Type:
  E{dw1: Array<F32>, dh: Array<F32>, x: Array<F32>, w1: Array<F32>, b1: Array<F32>, b2: Array<F32>, w2: Array<F32>, loss: F32, dw2: Array<F32>, db2: Array<F32>}

type FF is Type:
  FF{db1: Array<F32>, dw1: Array<F32>, w1: Array<F32>, w2: Array<F32>, b1: Array<F32>, b2: Array<F32>, loss: F32, dw2: Array<F32>, db2: Array<F32>}

type R is Type:
  R{w1: Array<F32>, b1: Array<F32>, w2: Array<F32>, b2: Array<F32>, loss: F32}

# --- chain -----------------------------------------------------------------

def f1.clone(pb: Array<F32> & Array<F32>, t3: T.T3, +batch: U32, +hid: U32) -> F1:
  match pb:
    case (b1a, b1b):
      match t3:
        case T.T3{h0, x, w1}:
          h = T.relu(U32.mul(batch, hid), bias_add(U32.mul(batch, hid), h0, b1a, hid))
          F1{h, x, w1, b1b}

def f1(t3: T.T3, b1: Array<F32>, +batch: U32, +hid: U32) -> F1:
  f1.clone(Array.clone(F32, b1), t3, batch, hid)

def f2.go(t3: T.T3, x: Array<F32>, w1: Array<F32>, b1: Array<F32>, b2a: Array<F32>, b2b: Array<F32>,
          +batch: U32, +cols: U32) -> F2:
  match t3:
    case T.T3{y0, h, w2}:
      F2{bias_add(U32.mul(batch, cols), y0, b2a, cols), h, w2, x, w1, b1, b2b}

def f2.clone(pb: Array<F32> & Array<F32>, f: F1, w2: Array<F32>,
             +batch: U32, +cols: U32, +hid: U32) -> F2:
  match pb:
    case (b2a, b2b):
      match f:
        case F1{h, x, w1, b1}:
          f2.go(T.matmul3(h, w2, Array.new(F32, U32.log2(U32.mul(batch, cols)), 0.0), batch, cols, hid), x, w1, b1, b2a, b2b, batch, cols)

def f2(f: F1, w2: Array<F32>, b2: Array<F32>, +batch: U32, +cols: U32, +hid: U32) -> F2:
  f2.clone(Array.clone(F32, b2), f, w2, batch, cols, hid)

def lb.done(l3: L3, h: Array<F32>, w2: Array<F32>, x: Array<F32>, w1: Array<F32>,
            b1: Array<F32>, b2: Array<F32>) -> LB:
  match l3:
    case L3{loss, y, t}:
      LB{loss, y, t, h, w2, x, w1, b1, b2}

def l(f: F2, t: Array<F32>, +batch: U32, +cols: U32) -> LB:
  match f:
    case F2{y, h, w2, x, w1, b1, b2}:
      lb.done(mse3(U32.mul(batch, cols), y, t), h, w2, x, w1, b1, b2)

def g(lb: LB, +batch: U32, +cols: U32) -> GB:
  match lb:
    case LB{loss, y, t, h, w2, x, w1, b1, b2}:
      GB{grad3(U32.mul(batch, cols), y, t), h, w2, x, w1, b1, b2, loss}

def a.go(t3: T.T3, w2: Array<F32>, x: Array<F32>, w1: Array<F32>, b1: Array<F32>, b2: Array<F32>, loss: F32) -> A:
  match t3:
    case T.T3{dw2, h2, dy2}:
      A{dw2, h2, dy2, w2, x, w1, b1, b2, loss}

def a(gb: GB, +batch: U32, +cols: U32, +hid: U32) -> A:
  match gb:
    case GB{dy, h, w2, x, w1, b1, b2, loss}:
      a.go(T.matmul3_tn(h, dy, Array.new(F32, U32.log2(U32.mul(hid, cols)), 0.0), hid, cols, batch), w2, x, w1, b1, b2, loss)

def b.go(t3: T.T3, h: Array<F32>, x: Array<F32>, w1: Array<F32>, b1: Array<F32>, b2: Array<F32>, loss: F32, dw2: Array<F32>) -> B:
  match t3:
    case T.T3{dh0, dy2, w2b}:
      B{dh0, h, dy2, w2b, x, w1, b1, b2, loss, dw2}

def b(a: A, +batch: U32, +cols: U32, +hid: U32) -> B:
  match a:
    case A{dw2, h, dy, w2, x, w1, b1, b2, loss}:
      b.go(T.matmul3_nt(dy, w2, Array.new(F32, U32.log2(U32.mul(batch, hid)), 0.0), batch, hid, cols), h, x, w1, b1, b2, loss, dw2)

def c(b: B, +batch: U32, +cols: U32) -> C:
  match b:
    case B{dh0, h, dy, w2, x, w1, b1, b2, loss, dw2}:
      C{colsum(batch, cols, dy), dh0, h, w2, x, w1, b1, b2, loss, dw2}

def d(c: C, +batch: U32, +hid: U32) -> D:
  match c:
    case C{db2, dh0, h, w2, x, w1, b1, b2, loss, dw2}:
      D{mask(U32.mul(batch, hid), dh0, h), w2, x, w1, b1, b2, loss, dw2, db2}

def e.go(t3: T.T3, w1: Array<F32>, b1: Array<F32>, b2: Array<F32>, w2: Array<F32>, loss: F32, dw2: Array<F32>, db2: Array<F32>) -> E:
  match t3:
    case T.T3{dw1, x2, dh2}:
      E{dw1, dh2, x2, w1, b1, b2, w2, loss, dw2, db2}

def e(d: D, +batch: U32, +depth: U32, +hid: U32) -> E:
  match d:
    case D{dh, w2, x, w1, b1, b2, loss, dw2, db2}:
      e.go(T.matmul3_tn(x, dh, Array.new(F32, U32.log2(U32.mul(depth, hid)), 0.0), depth, hid, batch), w1, b1, b2, w2, loss, dw2, db2)

def ff(e: E, +batch: U32, +hid: U32) -> FF:
  match e:
    case E{dw1, dh, x, w1, b1, b2, w2, loss, dw2, db2}:
      FF{colsum(batch, hid, dh), dw1, w1, w2, b1, b2, loss, dw2, db2}

def upd(f: FF, +cols: U32, +hid: U32, +depth: U32, +lr: F32) -> R:
  match f:
    case FF{db1, dw1, w1, w2, b1, b2, loss, dw2, db2}:
      R{sgd(U32.mul(depth, hid), w1, dw1, lr), sgd(hid, b1, db1, lr),
        sgd(U32.mul(hid, cols), w2, dw2, lr), sgd(cols, b2, db2, lr), loss}

def step(w1: Array<F32>, b1: Array<F32>, w2: Array<F32>, b2: Array<F32>,
         x: Array<F32>, t: Array<F32>, +batch: U32, +cols: U32, +depth: U32, +hid: U32, +lr: F32) -> R:
  upd(
    ff(
      e(
        d(
          c(
            b(
              a(
                g(
                  l(
                    f2(
                      f1(T.matmul3(x, w1, Array.new(F32, U32.log2(U32.mul(batch, hid)), 0.0), batch, hid, depth), b1, batch, hid),
                      w2, b2, batch, cols, hid),
                    t, batch, cols),
                  batch, cols),
                batch, cols, hid),
              batch, cols, hid),
            batch, cols),
          batch, hid),
        batch, depth, hid),
      batch, hid),
    cols, hid, depth, lr)

def epoch.go(n: Nat, r: R, +off: U32, +batch: U32, +cols: U32, +depth: U32, +hid: U32, +lr: F32) -> R:
  match n:
    case 0n:
      r
    case 1n++p:
      match r:
        case R{w1, b1, w2, b2, loss}:
          x = mkx(batch, off)
          t = mkt(batch, off)
          epoch.go(p, step(w1, b1, w2, b2, x, t, batch, cols, depth, hid, lr), U32.add(off, batch), batch, cols, depth, hid, lr)

def train.n(n: Nat, r: R, +batch: U32, +cols: U32, +depth: U32, +hid: U32, +lr: F32) -> R:
  epoch.go(n, r, 0, batch, cols, depth, hid, lr)

def lb.loss(lb: LB) -> F32:
  match lb:
    case LB{loss, y, t, h, w2, x, w1, b1, b2}:
      loss

def eval.r(r: R, +batch: U32, +cols: U32, +depth: U32, +hid: U32) -> F32:
  match r:
    case R{w1, b1, w2, b2, loss}:
      x = mkx(batch, 0)
      t = mkt(batch, 0)
      +cells = U32.mul(batch, cols)
      F32.div(lb.loss(l(f2(f1(T.matmul3(x, w1, Array.new(F32, U32.log2(U32.mul(batch, hid)), 0.0), batch, hid, depth), b1, batch, hid), w2, b2, batch, cols, hid), t, batch, cols)), U32.to_f32(cells))

def report(r: R, t0: Nat, t1: Nat) -> IO(Unit):
  do IO<Unit>:
    +l : F32 = eval.r(r, 64, 4, 2, 8)
    IO.print(F32.show(l) ++ "," ++ Nat.show(Nat.sub(t1, t0)))

def main() -> IO(Unit):
  do IO<Unit>:
    +batch : U32 = 64
    +cols : U32 = 4
    +depth : U32 = 2
    +hid : U32 = 8
    +lr : F32 = 1.0
    w1 : Array<F32> = T.map_scale(16, T.gen2(16, 7), 0.5)
    b1 : Array<F32> = Array.new(F32, 3n, 0.1)
    w2 : Array<F32> = T.map_scale(32, T.gen2(32, 9), 0.5)
    b2 : Array<F32> = Array.new(F32, 2n, 0.0)
    r0 : R = R{w1, b1, w2, b2, 0.0}
    t0 : Nat <- IO.now()
    r : R = train.n(2048n, r0, batch, cols, depth, hid, lr)
    t1 : Nat <- IO.now()
    report(r, t0, t1)
