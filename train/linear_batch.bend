import Base
import ../src/tensor.bend as T
import ../src/mlp.bend as M

# Batched linear readout: y = x Wt + b, MSE, full 64-sample mini-batches.
# Wt is D x C. The matmul path is T.matmul3 / T.matmul3_tn; every step
# returns the arrays the next step needs.

type L3 is Type:
  L3{loss: F32, y: Array<F32>, t: Array<F32>}

type F is Type:
  F{y: Array<F32>, x: Array<F32>, wt: Array<F32>}

type L is Type:
  L{loss: F32, y: Array<F32>, t: Array<F32>, x: Array<F32>, wt: Array<F32>}

type G is Type:
  G{dy: Array<F32>, x: Array<F32>, wt: Array<F32>, loss: F32}

type D is Type:
  D{dwt: Array<F32>, dy: Array<F32>, x: Array<F32>, wt: Array<F32>, loss: F32}

type DB is Type:
  DB{db: Array<F32>, dwt: Array<F32>, wt: Array<F32>, loss: F32}

type R is Type:
  R{wt: Array<F32>, b: Array<F32>, loss: F32}

# ---------------------------------------------------------------------------
# Array kernels
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# Chain
# ---------------------------------------------------------------------------

def fwd(t3: T.T3, b: Array<F32>, +batch: U32, +cols: U32) -> F:
  match t3:
    case T.T3{c, x, wt}:
      F{bias_add(U32.mul(batch, cols), c, b, cols), x, wt}

def loss.done(l3: L3, x: Array<F32>, wt: Array<F32>) -> L:
  match l3:
    case L3{loss, y, t}:
      L{loss, y, t, x, wt}

def loss(f: F, t: Array<F32>, +batch: U32, +cols: U32) -> L:
  match f:
    case F{y, x, wt}:
      loss.done(mse3(U32.mul(batch, cols), y, t), x, wt)

def grad(l: L, +batch: U32, +cols: U32) -> G:
  match l:
    case L{loss, y, t, x, wt}:
      G{grad3(U32.mul(batch, cols), y, t), x, wt, loss}

def dw.done(t3: T.T3, wt: Array<F32>, loss: F32) -> D:
  match t3:
    case T.T3{dwt, x, dy}:
      D{dwt, dy, x, wt, loss}

def dw(g: G, +batch: U32, +cols: U32, +depth: U32) -> D:
  match g:
    case G{dy, x, wt, loss}:
      dw.done(T.matmul3_tn(x, dy, Array.new(F32, U32.log2(U32.mul(cols, depth)), 0.0), depth, cols, batch), wt, loss)

def db(d: D, +batch: U32, +cols: U32) -> DB:
  match d:
    case D{dwt, dy, x, wt, loss}:
      DB{colsum(batch, cols, dy), dwt, wt, loss}

def upd(db: DB, b: Array<F32>, +cols: U32, +depth: U32, +lr: F32) -> R:
  match db:
    case DB{db, dwt, wt, loss}:
      R{sgd(U32.mul(cols, depth), wt, dwt, lr), sgd(cols, b, db, lr), loss}

def step.clone(pb: Array<F32> & Array<F32>, wt: Array<F32>, x: Array<F32>, t: Array<F32>,
               +batch: U32, +cols: U32, +depth: U32, +lr: F32) -> R:
  match pb:
    case (b1, b2):
      upd(
        db(
          dw(
            grad(
              loss(
                fwd(
                  T.matmul3(x, wt, Array.new(F32, U32.log2(U32.mul(batch, cols)), 0.0), batch, cols, depth),
                  b1, batch, cols),
                t, batch, cols),
              batch, cols),
            batch, cols, depth),
          batch, cols),
        b2, cols, depth, lr)

def step(wt: Array<F32>, b: Array<F32>, x: Array<F32>, t: Array<F32>,
         +batch: U32, +cols: U32, +depth: U32, +lr: F32) -> R:
  step.clone(Array.clone(F32, b), wt, x, t, batch, cols, depth, lr)

def epoch.go(n: Nat, r: R, +off: U32, +batch: U32, +cols: U32, +depth: U32, +lr: F32) -> R:
  match n:
    case 0n:
      r
    case 1n++p:
      match r:
        case R{wt, b, loss}:
          x = mkx(batch, off)
          t = mkt(batch, off)
          epoch.go(p, step(wt, b, x, t, batch, cols, depth, lr), U32.add(off, batch), batch, cols, depth, lr)

def train.n(n: Nat, r: R, +batch: U32, +cols: U32, +depth: U32, +lr: F32) -> R:
  epoch.go(n, r, 0, batch, cols, depth, lr)

def eval.b(t3: T.T3, b: Array<F32>, +batch: U32, +cols: U32) -> Array<F32>:
  match t3:
    case T.T3{y0, x2, wt2}:
      bias_add(U32.mul(batch, cols), y0, b, cols)

def eval.m(l3: L3, cells: U32) -> F32:
  match l3:
    case L3{s, y3, t3}:
      F32.div(s, U32.to_f32(cells))

def eval.r(r: R, +batch: U32, +cols: U32, +depth: U32) -> F32:
  match r:
    case R{wt, b, loss}:
      x = mkx(batch, 0)
      t = mkt(batch, 0)
      +cells = U32.mul(batch, cols)
      eval.m(
        mse3(cells,
          eval.b(T.matmul3(x, wt, Array.new(F32, U32.log2(cells), 0.0), batch, cols, depth), b, batch, cols),
          t),
        cells)

def report(r: R, t0: Nat, t1: Nat, +batch: U32, +cols: U32, +depth: U32) -> IO(Unit):
  do IO<Unit>:
    +l : F32 = eval.r(r, batch, cols, depth)
    IO.print(F32.show(l) ++ "," ++ Nat.show(Nat.sub(t1, t0)))

def main() -> IO(Unit):
  do IO<Unit>:
    +batch : U32 = 64
    +cols : U32 = 4
    +depth : U32 = 2
    +lr : F32 = 0.5
    wt : Array<F32> = T.map_scale(8, T.gen2(8, 7), 0.5)
    b : Array<F32> = Array.new(F32, 2n, 0.0)
    r0 : R = R{wt, b, 0.0}
    t0 : Nat <- IO.now()
    r : R = train.n(40n, r0, batch, cols, depth, lr)
    t1 : Nat <- IO.now()
    report(r, t0, t1, batch, cols, depth)
