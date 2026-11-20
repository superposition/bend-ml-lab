import Base
import ../src/mlp.bend as M

# Scalar baseline for the batched linear readout: per-sample MSE SGD,
# Wt is D x C, b is C.

type SLM is Data:
  SLM{wt: List<&2, F32>, b: List<&2, F32>, loss: F32}

def upd.wt(d: Nat, c: Nat, +wt: List<&2, F32>, +g: F32, +x0: F32, +x1: F32, +lr: F32) -> List<&2, F32>:
  match d:
    case 0n:
      +at = c
      +v = F32.sub(M.lget(wt, at), F32.mul(lr, F32.mul(g, x0)))
      M.lset(wt, at, v)
    case 1n+p:
      +at = Nat.add(4n, c)
      +v = F32.sub(M.lget(wt, at), F32.mul(lr, F32.mul(g, x1)))
      M.lset(wt, at, v)

def upd.c(n: Nat, +wt: List<&2, F32>, +b: List<&2, F32>, +yg: List<&2, F32>,
           +x0: F32, +x1: F32, +lr: F32) -> SLM:
  match n:
    case 0n:
      SLM{wt, b, 0.0}
    case 1n++p:
      +g = M.lget(yg, p)
      +wt1 = upd.wt(0n, p, wt, g, x0, x1, lr)
      +wt2 = upd.wt(1n, p, wt1, g, x0, x1, lr)
      +b2 = M.lset(b, p, F32.sub(M.lget(b, p), F32.mul(lr, g)))
      upd.c(p, wt2, b2, yg, x0, x1, lr)

def sstep(s: SLM, +i: U32, +lr: F32) -> SLM:
  match s:
    case SLM{+wt, +b, loss}:
      +x0 = M.coord(i, 1)
      +x1 = M.coord(i, 2)
      +label = U32.to_nat(U32.and(i, 3))
      +y0 = F32.add(F32.add(F32.mul(M.lget(wt, 0n), x0), F32.mul(M.lget(wt, 4n), x1)), M.lget(b, 0n))
      +y1 = F32.add(F32.add(F32.mul(M.lget(wt, 1n), x0), F32.mul(M.lget(wt, 5n), x1)), M.lget(b, 1n))
      +y2 = F32.add(F32.add(F32.mul(M.lget(wt, 2n), x0), F32.mul(M.lget(wt, 6n), x1)), M.lget(b, 2n))
      +y3 = F32.add(F32.add(F32.mul(M.lget(wt, 3n), x0), F32.mul(M.lget(wt, 7n), x1)), M.lget(b, 3n))
      +t0 = M.sel(1.0, 0.0, Nat.is_eq(0n, label))
      +t1 = M.sel(1.0, 0.0, Nat.is_eq(1n, label))
      +t2 = M.sel(1.0, 0.0, Nat.is_eq(2n, label))
      +t3 = M.sel(1.0, 0.0, Nat.is_eq(3n, label))
      +g0 = F32.mul(2.0, F32.sub(y0, t0))
      +g1 = F32.mul(2.0, F32.sub(y1, t1))
      +g2 = F32.mul(2.0, F32.sub(y2, t2))
      +g3 = F32.mul(2.0, F32.sub(y3, t3))
      +yg : List<&2, F32> = g0 <> g1 <> g2 <> g3 <> Nil{}
      upd.c(4n, wt, b, yg, x0, x1, lr)

def sep.go(n: Nat, +i: U32, +s: SLM, +lr: F32) -> SLM:
  match n:
    case 0n:
      s
    case 1n++p:
      sep.go(p, U32.add(i, 1), sstep(s, i, lr), lr)

def sep(s: SLM, +lr: F32) -> SLM:
  sep.go(256n, 0, s, lr)

def strain.n(n: Nat, +s: SLM, +lr: F32) -> SLM:
  match n:
    case 0n:
      s
    case 1n++p:
      strain.n(p, sep(s, lr), lr)

def ev.go(n: Nat, +i: U32, +s: SLM) -> F32:
  match n:
    case 0n:
      0.0
    case 1n++p:
      match s:
        case SLM{+wt, +b, loss}:
          +x0 = M.coord(i, 1)
          +x1 = M.coord(i, 2)
          +label = U32.to_nat(U32.and(i, 3))
          +y0 = F32.add(F32.add(F32.mul(M.lget(wt, 0n), x0), F32.mul(M.lget(wt, 4n), x1)), M.lget(b, 0n))
          +y1 = F32.add(F32.add(F32.mul(M.lget(wt, 1n), x0), F32.mul(M.lget(wt, 5n), x1)), M.lget(b, 1n))
          +y2 = F32.add(F32.add(F32.mul(M.lget(wt, 2n), x0), F32.mul(M.lget(wt, 6n), x1)), M.lget(b, 2n))
          +y3 = F32.add(F32.add(F32.mul(M.lget(wt, 3n), x0), F32.mul(M.lget(wt, 7n), x1)), M.lget(b, 3n))
          +t0 = M.sel(1.0, 0.0, Nat.is_eq(0n, label))
          +t1 = M.sel(1.0, 0.0, Nat.is_eq(1n, label))
          +t2 = M.sel(1.0, 0.0, Nat.is_eq(2n, label))
          +t3 = M.sel(1.0, 0.0, Nat.is_eq(3n, label))
          +d0 = F32.sub(y0, t0)
          +d1 = F32.sub(y1, t1)
          +d2 = F32.sub(y2, t2)
          +d3 = F32.sub(y3, t3)
          +cell = F32.add(F32.mul(d0, d0), F32.add(F32.mul(d1, d1), F32.add(F32.mul(d2, d2), F32.mul(d3, d3))))
          F32.add(F32.div(cell, 4.0), ev.go(p, U32.add(i, 1), s))

def ev(s: SLM) -> F32:
  F32.div(ev.go(256n, 0, s), 256.0)

def main() -> IO(Unit):
  do IO<Unit>:
    +lr : F32 = 0.01
    wt : List<&2, F32> = M.lscale(M.mk(8n, 7), 0.5)
    b : List<&2, F32> = M.mk(4n, 99)
    s0 : SLM = SLM{wt, b, 0.0}
    t0 : Nat <- IO.now()
    s : SLM = strain.n(40n, s0, lr)
    t1 : Nat <- IO.now()
    +l : F32 = ev(s)
    IO.print(F32.show(l) ++ "," ++ Nat.show(Nat.sub(t1, t0)))
