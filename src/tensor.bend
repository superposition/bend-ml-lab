import Base

# tensor.bend — flat, row-major F32 tensors for Bend 2.0.x.
#
# Storage is one `Array<F32>` padded to a power of two. Shapes are passed
# beside the array as U32 values and are expected to be powers of two
# (Bend's `Array.new` takes a depth, so 2^d slots). This is the least
# surprising representation that stays inside Bend's affine array rules:
# an Array<T> is a Type, exactly one owner, and every read hands the array
# back (`Array.get` returns `Array<T> & T`).
#
# Two Bend idioms recur below and are load-bearing:
#
# 1. A `match` may only scrutinize a parameter or a field, never a computed
#    value. A function that returns a tuple is therefore consumed by a def
#    that matches its parameter, not destructured where it is called.
# 2. Scrutinees are matched in binder order. `match fuel:` must come before
#    `match ar:` when `fuel` is the first parameter.

# ---------------------------------------------------------------------------
# Deterministic data
# ---------------------------------------------------------------------------

# xorshift32; same constants as the official Bend nbody bench.
def prng(+x: U32) -> U32:
  +b = U32.xor(x, U32.shln(x, 13n))
  +d = U32.xor(b, U32.shrn(b, 17n))
  U32.xor(d, U32.shln(d, 5n))

# Uniform [0, 1) from the top 24 bits.
def unit(+s: U32) -> F32:
  F32.div(U32.to_f32(U32.shrn(s, 8n)), U32.to_f32(16777216))

def fill(fuel: Nat, a: Array<F32>, +i: U32, seed: U32) -> Array<F32>:
  match fuel:
    case 0n:
      a
    case 1n++p:
      +s = prng(seed)
      fill(p, Array.set(F32, a, i, unit(s)), U32.add(i, 1), s)

# Deterministic matrix of `cells` entries (a power of two) from `seed`.
def gen(+cells: U32, seed: U32) -> Array<F32>:
  fill(U32.to_nat(cells), Array.new(F32, U32.log2(cells), 0.0), 0, seed)

# Uniform [-1, 1): used for weights.
def unit2(+s: U32) -> F32:
  F32.sub(F32.mul(unit(s), 2.0), 1.0)

def fill2(fuel: Nat, a: Array<F32>, +i: U32, seed: U32) -> Array<F32>:
  match fuel:
    case 0n:
      a
    case 1n++p:
      +s = prng(seed)
      fill2(p, Array.set(F32, a, i, unit2(s)), U32.add(i, 1), s)

def gen2(+cells: U32, seed: U32) -> Array<F32>:
  fill2(U32.to_nat(cells), Array.new(F32, U32.log2(cells), 0.0), 0, seed)

# ---------------------------------------------------------------------------
# Dot product / matmul
# ---------------------------------------------------------------------------

# dot: consume `fuel` pairs of values, stepping a by `astep` and b by
# `bstep`. `ar`/`br` hold the value at the current (ai, bi). The result
# returns the arrays after the last read so the caller can keep using them.
#
# One read past the logical end happens at fuel = 0; indexes wrap in
# Bend's arrays, so this is harmless and keeps the state machine flat.
def dot(
  fuel: Nat,
  ar: Array<F32> & F32,
  br: Array<F32> & F32,
  +ai: U32,
  +bi: U32,
  +astep: U32,
  +bstep: U32,
  acc: F32
) -> Array<F32> & Array<F32> & F32:
  match fuel:
    case 0n:
      match ar:
        case (a1, x):
          match br:
            case (b1, y):
              (a1, b1, acc)
    case 1n++p:
      match ar:
        case (a1, x):
          match br:
            case (b1, y):
              +an = U32.add(ai, astep)
              +bn = U32.add(bi, bstep)
              dot(
                p,
                Array.get(F32, a1, an),
                Array.get(F32, b1, bn),
                an,
                bn,
                astep,
                bstep,
                F32.add(acc, F32.mul(x, y))
              )

# Matrix product C[M,N] = A[M,K] * B[K,N], row-major.
#
# `dr` is the dot result for the current output cell (the pair carries the
# array handles forward). One extra dot is computed at the end of a range;
# at M*N per matmul it is noise.
def mm(
  fuel: Nat,
  dr: Array<F32> & Array<F32> & F32,
  c: Array<F32>,
  +m: U32,
  +n: U32,
  +k: U32,
  +row: U32,
  +col: U32
) -> Array<F32>:
  match fuel:
    case 0n:
      c
    case 1n+q:
      match dr:
        case (a1, b1, acc):
          p = U32.add(U32.mul(row, n), col)
          c1 = Array.set(F32, c, p, acc)
          +col1 = U32.add(col, 1)
          +wrap = U32.is_eq(col1, n)
          +row1 = U32.add(row, Bool.to_u32(wrap))
          +col2 = U32.sub(col1, U32.mul(n, Bool.to_u32(wrap)))
          +ai = U32.mul(row1, k)
          mm(
            q,
            dot(
              U32.to_nat(k),
              Array.get(F32, a1, ai),
              Array.get(F32, b1, col2),
              ai,
              col2,
              1,
              n,
              0.0
            ),
            c1,
            m,
            n,
            k,
            row1,
            col2
          )

# Seed mm for a fresh C.  `a`, `b`, `c` must be sized to their padded
# shapes; m, n and k are powers of two.
def matmul(a: Array<F32>, b: Array<F32>, c: Array<F32>, +m: U32, +n: U32, +k: U32) -> Array<F32>:
  mm(
    U32.to_nat(U32.mul(m, n)),
    dot(
      U32.to_nat(k),
      Array.get(F32, a, 0),
      Array.get(F32, b, 0),
      0,
      0,
      1,
      n,
      0.0
    ),
    c,
    m,
    n,
    k,
    0,
    0
  )


# C[M,N] = A[K,M]^T * B[K,N]: both inputs are stored transposed.
def mm_tn(
  fuel: Nat,
  dr: Array<F32> & Array<F32> & F32,
  c: Array<F32>,
  +m: U32,
  +n: U32,
  +k: U32,
  +row: U32,
  +col: U32
) -> Array<F32>:
  match fuel:
    case 0n:
      c
    case 1n+q:
      match dr:
        case (a1, b1, acc):
          p = U32.add(U32.mul(row, n), col)
          c1 = Array.set(F32, c, p, acc)
          +col1 = U32.add(col, 1)
          +wrap = U32.is_eq(col1, n)
          +row1 = U32.add(row, Bool.to_u32(wrap))
          +col2 = U32.sub(col1, U32.mul(n, Bool.to_u32(wrap)))
          mm_tn(
            q,
            dot(
              U32.to_nat(k),
              Array.get(F32, a1, row1),
              Array.get(F32, b1, col2),
              row1,
              col2,
              m,
              n,
              0.0
            ),
            c1,
            m,
            n,
            k,
            row1,
            col2
          )

def matmul_tn(a: Array<F32>, b: Array<F32>, c: Array<F32>, +m: U32, +n: U32, +k: U32) -> Array<F32>:
  mm_tn(
    U32.to_nat(U32.mul(m, n)),
    dot(
      U32.to_nat(k),
      Array.get(F32, a, 0),
      Array.get(F32, b, 0),
      0,
      0,
      m,
      n,
      0.0
    ),
    c,
    m,
    n,
    k,
    0,
    0
  )

# Parallel matmul (CPU fork/join, `!` for the GPU) lives in parallel.bend:
# it is the only @unsafe module, so the safe core can pass --check-only.

# ---------------------------------------------------------------------------
# Elementwise / reductions
# ---------------------------------------------------------------------------

def sum(fuel: Nat, ar: Array<F32> & F32, +i: U32, acc: F32) -> Array<F32> & F32:
  match fuel:
    case 0n:
      match ar:
        case (a1, x):
          (a1, acc)
    case 1n++p:
      match ar:
        case (a1, x):
          sum(p, Array.get(F32, a1, U32.add(i, 1)), U32.add(i, 1), F32.add(acc, x))

def csum(fuel: Nat, ar: Array<F32> & F32, +i: U32, acc: F32) -> F32:
  match fuel:
    case 0n:
      match ar:
        case (a1, x):
          acc
    case 1n++p:
      match ar:
        case (a1, x):
          csum(p, Array.get(F32, a1, U32.add(i, 1)), U32.add(i, 1), F32.add(acc, x))

# Sum of all `cells` entries, used as a result checksum everywhere.
def checksum(cells: U32, a: Array<F32>) -> F32:
  csum(U32.to_nat(cells), Array.get(F32, a, 0), 0, 0.0)

def relu.sel(x: F32, neg: Bool) -> F32:
  match neg:
    case True{}:
      0.0
    case False{}:
      x

def relu.one(+x: F32) -> F32:
  relu.sel(x, F32.is_lt(x, 0.0))

def relu.go(fuel: Nat, ar: Array<F32> & F32, +i: U32) -> Array<F32>:
  match fuel:
    case 0n:
      match ar:
        case (a1, x):
          a1
    case 1n++p:
      match ar:
        case (a1, x):
          relu.go(p, Array.get(F32, Array.set(F32, a1, i, relu.one(x)), U32.add(i, 1)), U32.add(i, 1))

def relu(cells: U32, a: Array<F32>) -> Array<F32>:
  relu.go(U32.to_nat(cells), Array.get(F32, a, 0), 0)

def map_scale.go(fuel: Nat, ar: Array<F32> & F32, +i: U32, +s: F32) -> Array<F32>:
  match fuel:
    case 0n:
      match ar:
        case (a1, x):
          a1
    case 1n++p:
      match ar:
        case (a1, x):
          map_scale.go(p, Array.get(F32, Array.set(F32, a1, i, F32.mul(x, s)), U32.add(i, 1)), U32.add(i, 1), s)

def map_scale(cells: U32, a: Array<F32>, s: F32) -> Array<F32>:
  map_scale.go(U32.to_nat(cells), Array.get(F32, a, 0), 0, s)

def axpy.go(fuel: Nat, ar: Array<F32> & F32, br: Array<F32> & F32, +i: U32, +s: F32) -> Array<F32> & Array<F32>:
  match fuel:
    case 0n:
      match ar:
        case (a1, x):
          match br:
            case (b1, y):
              (a1, b1)
    case 1n++p:
      match ar:
        case (a1, x):
          match br:
            case (b1, y):
              +z = F32.add(x, F32.mul(s, y))
              +an = U32.add(i, 1)
              axpy.go(p, Array.get(F32, Array.set(F32, a1, i, z), an), Array.get(F32, b1, an), an, s)

def axpy(cells: U32, a: Array<F32>, b: Array<F32>, s: F32) -> Array<F32> & Array<F32>:
  axpy.go(U32.to_nat(cells), Array.get(F32, a, 0), Array.get(F32, b, 0), 0, s)

# ---------------------------------------------------------------------------
# Einsum primitives that are not matmul
# ---------------------------------------------------------------------------

# C[M,N] = a[M] outer b[N], row-major.
def outer.go(
  fuel: Nat,
  ar: Array<F32> & F32,
  br: Array<F32> & F32,
  c: Array<F32>,
  +n: U32,
  +p: U32
) -> Array<F32>:
  match fuel:
    case 0n:
      c
    case 1n+q:
      match ar:
        case (a1, x):
          match br:
            case (b1, y):
              c1 = Array.set(F32, c, p, F32.mul(x, y))
              +np = U32.add(p, 1)
              +nr = U32.div(np, n)
              +ns = U32.sub(np, U32.mul(nr, n))
              outer.go(q, Array.get(F32, a1, nr), Array.get(F32, b1, ns), c1, n, np)

def outer(+n: U32, a: Array<F32>, b: Array<F32>, c: Array<F32>) -> Array<F32>:
  outer.go(
    U32.to_nat(U32.mul(n, n)),
    Array.get(F32, a, 0),
    Array.get(F32, b, 0),
    c,
    n,
    0
  )

# Dot product of two flat vectors of `k` entries.
def vdot.go(fuel: Nat, ar: Array<F32> & F32, br: Array<F32> & F32, +i: U32, acc: F32) -> F32:
  match fuel:
    case 0n:
      acc
    case 1n+p:
      match ar:
        case (a1, x):
          match br:
            case (b1, y):
              +j = U32.add(i, 1)
              vdot.go(p, Array.get(F32, a1, j), Array.get(F32, b1, j), j, F32.add(acc, F32.mul(x, y)))

def vdot(k: U32, a: Array<F32>, b: Array<F32>) -> F32:
  vdot.go(U32.to_nat(k), Array.get(F32, a, 0), Array.get(F32, b, 0), 0, 0.0)

# out[j] = a[0,j] + a[1,j] + ... for an M x N row-major array.
def rowsum.go(
  fuel: Nat,
  ar: Array<F32> & F32,
  orr: Array<F32> & F32,
  +n: U32,
  +p: U32
) -> Array<F32>:
  match fuel:
    case 0n:
      match orr:
        case (o1, y):
          o1
    case 1n+q:
      match ar:
        case (a1, x):
          match orr:
            case (o1, y):
              j = U32.mod(p, n)
              o2 = Array.set(F32, o1, j, F32.add(y, x))
              +np = U32.add(p, 1)
              rowsum.go(q, Array.get(F32, a1, np), Array.get(F32, o2, U32.mod(np, n)), n, np)

def rowsum(m: U32, +n: U32, a: Array<F32>, out: Array<F32>) -> Array<F32>:
  rowsum.go(
    U32.to_nat(U32.mul(m, n)),
    Array.get(F32, a, 0),
    Array.get(F32, out, 0),
    n,
    0
  )

# Debug: print an array as [a, b, ...].
def showA(a: Array<F32>) -> String:
  Array.show(~F32, ~(s => (x => s ++ F32.show(x))), a)

# ---------------------------------------------------------------------------
# GPU-friendly variants: contiguous inner loops
# ---------------------------------------------------------------------------

# out[j*M + i] = a[i*N + j] for an M x N row-major array.
def transpose.go(fuel: Nat, ar: Array<F32> & F32, out: Array<F32>, +m: U32, +n: U32, +p: U32) -> Array<F32>:
  match fuel:
    case 0n:
      match ar:
        case (a1, x):
          out
    case 1n+q:
      match ar:
        case (a1, x):
          +i = U32.div(p, n)
          +j = U32.sub(p, U32.mul(i, n))
          +qat = U32.add(U32.mul(j, m), i)
          +np = U32.add(p, 1)
          transpose.go(q, Array.get(F32, a1, np), Array.set(F32, out, qat, x), m, n, np)

def transpose(+m: U32, +n: U32, a: Array<F32>, out: Array<F32>) -> Array<F32>:
  transpose.go(U32.to_nat(U32.mul(m, n)), Array.get(F32, a, 0), out, m, n, 0)

# C[M,N] = A[M,K] * Bt[N,K]^T. Both operands advance by 1 along K, which is
# the layout the GPU wants: one contiguous run per output cell, not a column
# stride of N.
def mm_bt(
  fuel: Nat,
  dr: Array<F32> & Array<F32> & F32,
  c: Array<F32>,
  +m: U32,
  +n: U32,
  +k: U32,
  +row: U32,
  +col: U32
) -> Array<F32>:
  match fuel:
    case 0n:
      c
    case 1n+q:
      match dr:
        case (a1, b1, acc):
          p = U32.add(U32.mul(row, n), col)
          c1 = Array.set(F32, c, p, acc)
          +col1 = U32.add(col, 1)
          +wrap = U32.is_eq(col1, n)
          +row1 = U32.add(row, Bool.to_u32(wrap))
          +col2 = U32.sub(col1, U32.mul(n, Bool.to_u32(wrap)))
          +ai = U32.mul(row1, k)
          +bi = U32.mul(col2, k)
          mm_bt(
            q,
            dot(
              U32.to_nat(k),
              Array.get(F32, a1, ai),
              Array.get(F32, b1, bi),
              ai,
              bi,
              1,
              1,
              0.0
            ),
            c1,
            m,
            n,
            k,
            row1,
            col2
          )

def matmul_bt(a: Array<F32>, bt: Array<F32>, c: Array<F32>, +m: U32, +n: U32, +k: U32) -> Array<F32>:
  mm_bt(
    U32.to_nat(U32.mul(m, n)),
    dot(
      U32.to_nat(k),
      Array.get(F32, a, 0),
      Array.get(F32, bt, 0),
      0,
      0,
      1,
      1,
      0.0
    ),
    c,
    m,
    n,
    k,
    0,
    0
  )

# A rectangular range of C: rows [row0, row0+rows), cols [col0, col0+cols).
def mm_bt_range(
  fuel: Nat,
  dr: Array<F32> & Array<F32> & F32,
  c: Array<F32>,
  +m: U32,
  +n: U32,
  +k: U32,
  +row: U32,
  +col: U32,
  +col0: U32,
  +col_end: U32
) -> Array<F32>:
  match fuel:
    case 0n:
      c
    case 1n+q:
      match dr:
        case (a1, b1, acc):
          p = U32.add(U32.mul(row, n), col)
          c1 = Array.set(F32, c, p, acc)
          +col1 = U32.add(col, 1)
          +wrap = U32.is_eq(col1, col_end)
          +row1 = U32.add(row, Bool.to_u32(wrap))
          +col2 = U32.sub(col1, U32.mul(U32.sub(col_end, col0), Bool.to_u32(wrap)))
          +ai = U32.mul(row1, k)
          +bi = U32.mul(col2, k)
          mm_bt_range(
            q,
            dot(
              U32.to_nat(k),
              Array.get(F32, a1, ai),
              Array.get(F32, b1, bi),
              ai,
              bi,
              1,
              1,
              0.0
            ),
            c1,
            m,
            n,
            k,
            row1,
            col2,
            col0,
            col_end
          )

# C rows [row0,row0+rows) x cols [col0,col0+cols) = A * Bt^T.
def matmul_bt_range(
  rows: U32,
  +cols: U32,
  a: Array<F32>,
  bt: Array<F32>,
  c: Array<F32>,
  +n: U32,
  +k: U32,
  +row0: U32,
  +col0: U32
) -> Array<F32>:
  +ai = U32.mul(row0, k)
  +bi = U32.mul(col0, k)
  mm_bt_range(
    U32.to_nat(U32.mul(rows, cols)),
    dot(
      U32.to_nat(k),
      Array.get(F32, a, ai),
      Array.get(F32, bt, bi),
      ai,
      bi,
      1,
      1,
      0.0
    ),
    c,
    n,
    n,
    k,
    row0,
    col0,
    col0,
    U32.add(col0, cols)
  )

# ---------------------------------------------------------------------------
# 2x2 tiled kernel (for the 2D tree, rows per lane = 2)
# ---------------------------------------------------------------------------

type Quad is Data:
  Q{a00: F32, a01: F32, a10: F32, a11: F32}

type KS is Type:
  KS{a0: Array<F32>, a1: Array<F32>, b0: Array<F32>, b1: Array<F32>, q: Quad}

type WS is Type:
  WS{c: Array<F32>, ks: KS}

def kq(
  fuel: Nat,
  ar0: Array<F32> & F32,
  ar1: Array<F32> & F32,
  br0: Array<F32> & F32,
  br1: Array<F32> & F32,
  q: Quad,
  +i0: U32,
  +i1: U32,
  +j0: U32,
  +j1: U32
) -> KS:
  match fuel:
    case 0n:
      match ar0:
        case (a0, x0):
          match ar1:
            case (a1, x1):
              match br0:
                case (b0, y0):
                  match br1:
                    case (b1, y1):
                      KS{a0, a1, b0, b1, q}
    case 1n+p:
      match ar0:
        case (a0, +x0):
          match ar1:
            case (a1, +x1):
              match br0:
                case (b0, +y0):
                  match br1:
                    case (b1, +y1):
                      match q:
                        case Q{+q00, +q01, +q10, +q11}:
                          +i0n = U32.add(i0, 1)
                          +i1n = U32.add(i1, 1)
                          +j0n = U32.add(j0, 1)
                          +j1n = U32.add(j1, 1)
                          kq(
                            p,
                            Array.get(F32, a0, i0n),
                            Array.get(F32, a1, i1n),
                            Array.get(F32, b0, j0n),
                            Array.get(F32, b1, j1n),
                            Q{
                              F32.add(q00, F32.mul(x0, y0)),
                              F32.add(q01, F32.mul(x0, y1)),
                              F32.add(q10, F32.mul(x1, y0)),
                              F32.add(q11, F32.mul(x1, y1))
                            },
                            i0n,
                            i1n,
                            j0n,
                            j1n
                          )

def kq.done(r: KS, c: Array<F32>, +row0: U32, +row1: U32, +col0: U32, +col1: U32, +n: U32) -> WS:
  match r:
    case KS{a0, a1, b0, b1, q}:
      match q:
        case Q{+q00, +q01, +q10, +q11}:
          c1 = Array.set(F32, c, U32.add(U32.mul(row0, n), col0), q00)
          c2 = Array.set(F32, c1, U32.add(U32.mul(row0, n), col1), q01)
          c3 = Array.set(F32, c2, U32.add(U32.mul(row1, n), col0), q10)
          c4 = Array.set(F32, c3, U32.add(U32.mul(row1, n), col1), q11)
          WS{c4, KS{a0, a1, b0, b1, Q{q00, q01, q10, q11}}}

def tile(
  k: Nat,
  a0: Array<F32>,
  a1: Array<F32>,
  b0: Array<F32>,
  b1: Array<F32>,
  c: Array<F32>,
  +row0: U32,
  +row1: U32,
  +col0: U32,
  +col1: U32,
  +n: U32
) -> WS:
  +i0 = U32.mul(row0, n)
  +i1 = U32.mul(row1, n)
  +j0 = U32.mul(col0, n)
  +j1 = U32.mul(col1, n)
  kq.done(
    kq(
      k,
      Array.get(F32, a0, i0),
      Array.get(F32, a1, i1),
      Array.get(F32, b0, j0),
      Array.get(F32, b1, j1),
      Q{0.0, 0.0, 0.0, 0.0},
      i0,
      i1,
      j0,
      j1
    ),
    c,
    row0,
    row1,
    col0,
    col1,
    n
  )

def tiles(
  fuel: Nat,
  w: WS,
  +row0: U32,
  +row1: U32,
  +col0: U32,
  +n: U32,
  +k: Nat
) -> WS:
  match fuel:
    case 0n:
      w
    case 1n+q:
      match w:
        case WS{c, ks}:
          match ks:
            case KS{a0, a1, b0, b1, _}:
              +col1 = U32.add(col0, 1)
              +col2 = U32.add(col0, 2)
              tiles(
                q,
                tile(k, a0, a1, b0, b1, c, row0, row1, col0, col1, n),
                row0,
                row1,
                col2,
                n,
                k
              )

def tiles.start(
  cols: U32,
  a0: Array<F32>,
  a1: Array<F32>,
  b0: Array<F32>,
  b1: Array<F32>,
  c: Array<F32>,
  +row0: U32,
  +col0: U32,
  +n: U32,
  +k: Nat
) -> WS:
  tiles(
    U32.to_nat(U32.shrn(cols, 1n)),
    WS{c, KS{a0, a1, b0, b1, Q{0.0, 0.0, 0.0, 0.0}}},
    row0,
    U32.add(row0, 1),
    col0,
    n,
    k
  )

def tile.done(w: WS) -> Array<F32>:
  match w:
    case WS{c, ks}:
      c


# ---------------------------------------------------------------------------
# 4x4 tiled kernel (rows per lane = 4)
# ---------------------------------------------------------------------------

type Hex is Data:
  H{h00: F32, h01: F32, h02: F32, h03: F32, h10: F32, h11: F32, h12: F32, h13: F32, h20: F32, h21: F32, h22: F32, h23: F32, h30: F32, h31: F32, h32: F32, h33: F32}

type K4 is Type:
  K{a0: Array<F32>, a1: Array<F32>, a2: Array<F32>, a3: Array<F32>,
    b0: Array<F32>, b1: Array<F32>, b2: Array<F32>, b3: Array<F32>, h: Hex}

type W4 is Type:
  W{c: Array<F32>, ks: K4}

def k4(
  fuel: Nat,
  ar0: Array<F32> & F32,
  ar1: Array<F32> & F32,
  ar2: Array<F32> & F32,
  ar3: Array<F32> & F32,
  br0: Array<F32> & F32,
  br1: Array<F32> & F32,
  br2: Array<F32> & F32,
  br3: Array<F32> & F32,
  h: Hex,
  +i0: U32,
  +i1: U32,
  +i2: U32,
  +i3: U32,
  +j0: U32
  +j1: U32
  +j2: U32
  +j3: U32
) -> K4:
  match fuel:
    case 0n:
      match ar0:
        case (a0, _):
          match ar1:
            case (a1, _):
              match ar2:
                case (a2, _):
                  match ar3:
                    case (a3, _):
                      match br0:
                        case (b0, _):
                          match br1:
                            case (b1, _):
                              match br2:
                                case (b2, _):
                                  match br3:
                                    case (b3, _):
                                      K{a0, a1, a2, a3, b0, b1, b2, b3, h}
    case 1n+p:
      match ar0:
        case (a0, +x0):
          match ar1:
            case (a1, +x1):
              match ar2:
                case (a2, +x2):
                  match ar3:
                    case (a3, +x3):
                      match br0:
                        case (b0, +y0):
                          match br1:
                            case (b1, +y1):
                              match br2:
                                case (b2, +y2):
                                  match br3:
                                    case (b3, +y3):
                                      match h:
                                        case H{h00, h01, h02, h03, h10, h11, h12, h13, h20, h21, h22, h23, h30, h31, h32, h33}:
                                          +i0n = U32.add(i0, 1)
                                          +i1n = U32.add(i1, 1)
                                          +i2n = U32.add(i2, 1)
                                          +i3n = U32.add(i3, 1)
                                          +j0n = U32.add(j0, 1)
                                          +j1n = U32.add(j1, 1)
                                          +j2n = U32.add(j2, 1)
                                          +j3n = U32.add(j3, 1)
                                          k4(
                                            p,
                                            Array.get(F32, a0, i0n),
                                            Array.get(F32, a1, i1n),
                                            Array.get(F32, a2, i2n),
                                            Array.get(F32, a3, i3n),
                                            Array.get(F32, b0, j0n),
                                            Array.get(F32, b1, j1n),
                                            Array.get(F32, b2, j2n),
                                            Array.get(F32, b3, j3n),
                                            H{F32.add(h00, F32.mul(x0, y0)), F32.add(h01, F32.mul(x0, y1)), F32.add(h02, F32.mul(x0, y2)), F32.add(h03, F32.mul(x0, y3)), F32.add(h10, F32.mul(x1, y0)), F32.add(h11, F32.mul(x1, y1)), F32.add(h12, F32.mul(x1, y2)), F32.add(h13, F32.mul(x1, y3)), F32.add(h20, F32.mul(x2, y0)), F32.add(h21, F32.mul(x2, y1)), F32.add(h22, F32.mul(x2, y2)), F32.add(h23, F32.mul(x2, y3)), F32.add(h30, F32.mul(x3, y0)), F32.add(h31, F32.mul(x3, y1)), F32.add(h32, F32.mul(x3, y2)), F32.add(h33, F32.mul(x3, y3))},
                                            i0n,
                                            i1n,
                                            i2n,
                                            i3n,
                                            j0n,
                                            j1n,
                                            j2n,
                                            j3n
                                          )

def k4.done(r: K4, c: Array<F32>, +row0: U32, +row1: U32, +row2: U32, +row3: U32,
            +col0: U32, +col1: U32, +col2: U32, +col3: U32, +n: U32) -> W4:
  match r:
    case K{a0, a1, a2, a3, b0, b1, b2, b3, h}:
      match h:
        case H{+h00, +h01, +h02, +h03, +h10, +h11, +h12, +h13, +h20, +h21, +h22, +h23, +h30, +h31, +h32, +h33}:
          c1 = Array.set(F32, c, U32.add(U32.mul(row0, n), col0), h00)
          c2 = Array.set(F32, c1, U32.add(U32.mul(row0, n), col1), h01)
          c3 = Array.set(F32, c2, U32.add(U32.mul(row0, n), col2), h02)
          c4 = Array.set(F32, c3, U32.add(U32.mul(row0, n), col3), h03)
          c5 = Array.set(F32, c4, U32.add(U32.mul(row1, n), col0), h10)
          c6 = Array.set(F32, c5, U32.add(U32.mul(row1, n), col1), h11)
          c7 = Array.set(F32, c6, U32.add(U32.mul(row1, n), col2), h12)
          c8 = Array.set(F32, c7, U32.add(U32.mul(row1, n), col3), h13)
          c9 = Array.set(F32, c8, U32.add(U32.mul(row2, n), col0), h20)
          c10 = Array.set(F32, c9, U32.add(U32.mul(row2, n), col1), h21)
          c11 = Array.set(F32, c10, U32.add(U32.mul(row2, n), col2), h22)
          c12 = Array.set(F32, c11, U32.add(U32.mul(row2, n), col3), h23)
          c13 = Array.set(F32, c12, U32.add(U32.mul(row3, n), col0), h30)
          c14 = Array.set(F32, c13, U32.add(U32.mul(row3, n), col1), h31)
          c15 = Array.set(F32, c14, U32.add(U32.mul(row3, n), col2), h32)
          c16 = Array.set(F32, c15, U32.add(U32.mul(row3, n), col3), h33)
          W{c16, K{a0, a1, a2, a3, b0, b1, b2, b3, H{h00, h01, h02, h03, h10, h11, h12, h13, h20, h21, h22, h23, h30, h31, h32, h33}}}

def tile4(k: Nat, a0: Array<F32>, a1: Array<F32>, a2: Array<F32>, a3: Array<F32>,
          b0: Array<F32>, b1: Array<F32>, b2: Array<F32>, b3: Array<F32>,
          c: Array<F32>, +row0: U32, +col0: U32, +n: U32) -> W4:
  +row1 = U32.add(row0, 1)
  +row2 = U32.add(row0, 2)
  +row3 = U32.add(row0, 3)
  +col1 = U32.add(col0, 1)
  +col2 = U32.add(col0, 2)
  +col3 = U32.add(col0, 3)
  +i0 = U32.mul(row0, n)
  +i1 = U32.mul(row1, n)
  +i2 = U32.mul(row2, n)
  +i3 = U32.mul(row3, n)
  +j0 = U32.mul(col0, n)
  +j1 = U32.mul(col1, n)
  +j2 = U32.mul(col2, n)
  +j3 = U32.mul(col3, n)
  k4.done(
    k4(
      k,
      Array.get(F32, a0, i0),
      Array.get(F32, a1, i1),
      Array.get(F32, a2, i2),
      Array.get(F32, a3, i3),
      Array.get(F32, b0, j0),
      Array.get(F32, b1, j1),
      Array.get(F32, b2, j2),
      Array.get(F32, b3, j3),
      H{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
      i0,
      i1,
      i2,
      i3,
      j0,
      j1,
      j2,
      j3
    ),
    c, row0, row1, row2, row3, col0, col1, col2, col3, n
  )

def tiles4(fuel: Nat, w: W4, +row0: U32, +col0: U32, +n: U32, +k: Nat) -> W4:
  match fuel:
    case 0n:
      w
    case 1n+q:
      match w:
        case W{c, ks}:
          match ks:
            case K{a0, a1, a2, a3, b0, b1, b2, b3, _}:
              +col4 = U32.add(col0, 4)
              tiles4(
                q,
                tile4(k, a0, a1, a2, a3, b0, b1, b2, b3, c, row0, col0, n),
                row0, col4, n, k
              )

def tiles4.start(cols: U32, a0: Array<F32>, a1: Array<F32>, a2: Array<F32>, a3: Array<F32>,
                 b0: Array<F32>, b1: Array<F32>, b2: Array<F32>, b3: Array<F32>,
                 c: Array<F32>, +row0: U32, +col0: U32, +n: U32, +k: Nat) -> W4:
  tiles4(
    U32.to_nat(U32.shrn(cols, 2n)),
    W{c, K{a0, a1, a2, a3, b0, b1, b2, b3,
      H{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}}},
    row0, col0, n, k
  )

def tile4.done(w: W4) -> Array<F32>:
  match w:
    case W{c, ks}:
      c

# Structural fold over the array tree: visits ALeaf/ANode, no index math.
def sum_tree(a: Array<F32>, acc: F32) -> F32:
  match a:
    case ALeaf{x}:
      F32.add(acc, x)
    case ANode{xs, ys}:
      sum_tree(ys, sum_tree(xs, acc))

# Fold over a copyable list (the representation Array.to_list produces).
def sum_list(xs: List<&1, F32>, acc: F32) -> F32:
  match xs:
    case Nil{}:
      acc
    case h <> t:
      sum_list(t, F32.add(acc, h))

# FNV-1a over the raw F32 bits, for cross-backend comparison.
def fnv(fuel: Nat, ar: Array<F32> & F32, +i: U32, h: U32) -> U32:
  match fuel:
    case 0n:
      h
    case 1n+p:
      match ar:
        case (a1, x):
          +b = F32.bits(x)
          +h2 = U32.mul(U32.xor(h, b), 16777619)
          fnv(p, Array.get(F32, a1, U32.add(i, 1)), U32.add(i, 1), h2)

def hash_bits(cells: U32, a: Array<F32>) -> U32:
  fnv(U32.to_nat(cells), Array.get(F32, a, 0), 0, 2166136261)

# ---------------------------------------------------------------------------
# Matmul variants that return their inputs (for batched training)
# ---------------------------------------------------------------------------

type T3 is Type:
  T3{c: Array<F32>, a: Array<F32>, b: Array<F32>}

def mm3(
  fuel: Nat,
  dr: Array<F32> & Array<F32> & F32,
  c: Array<F32>,
  +m: U32,
  +n: U32,
  +k: U32,
  +row: U32,
  +col: U32
) -> T3:
  match fuel:
    case 0n:
      match dr:
        case (a1, b1, acc):
          T3{c, a1, b1}
    case 1n+q:
      match dr:
        case (a1, b1, acc):
          p = U32.add(U32.mul(row, n), col)
          c1 = Array.set(F32, c, p, acc)
          +col1 = U32.add(col, 1)
          +wrap = U32.is_eq(col1, n)
          +row1 = U32.add(row, Bool.to_u32(wrap))
          +col2 = U32.sub(col1, U32.mul(n, Bool.to_u32(wrap)))
          +ai = U32.mul(row1, k)
          mm3(
            q,
            dot(
              U32.to_nat(k),
              Array.get(F32, a1, ai),
              Array.get(F32, b1, col2),
              ai,
              col2,
              1,
              n,
              0.0
            ),
            c1,
            m,
            n,
            k,
            row1,
            col2
          )

def matmul3(a: Array<F32>, b: Array<F32>, c: Array<F32>, +m: U32, +n: U32, +k: U32) -> T3:
  mm3(
    U32.to_nat(U32.mul(m, n)),
    dot(
      U32.to_nat(k),
      Array.get(F32, a, 0),
      Array.get(F32, b, 0),
      0,
      0,
      1,
      n,
      0.0
    ),
    c,
    m,
    n,
    k,
    0,
    0
  )

def mm3_tn(
  fuel: Nat,
  dr: Array<F32> & Array<F32> & F32,
  c: Array<F32>,
  +m: U32,
  +n: U32,
  +k: U32,
  +row: U32,
  +col: U32
) -> T3:
  match fuel:
    case 0n:
      match dr:
        case (a1, b1, acc):
          T3{c, a1, b1}
    case 1n+q:
      match dr:
        case (a1, b1, acc):
          p = U32.add(U32.mul(row, n), col)
          c1 = Array.set(F32, c, p, acc)
          +col1 = U32.add(col, 1)
          +wrap = U32.is_eq(col1, n)
          +row1 = U32.add(row, Bool.to_u32(wrap))
          +col2 = U32.sub(col1, U32.mul(n, Bool.to_u32(wrap)))
          mm3_tn(
            q,
            dot(
              U32.to_nat(k),
              Array.get(F32, a1, row1),
              Array.get(F32, b1, col2),
              row1,
              col2,
              m,
              n,
              0.0
            ),
            c1,
            m,
            n,
            k,
            row1,
            col2
          )

def matmul3_tn(a: Array<F32>, b: Array<F32>, c: Array<F32>, +m: U32, +n: U32, +k: U32) -> T3:
  mm3_tn(
    U32.to_nat(U32.mul(m, n)),
    dot(
      U32.to_nat(k),
      Array.get(F32, a, 0),
      Array.get(F32, b, 0),
      0,
      0,
      m,
      n,
      0.0
    ),
    c,
    m,
    n,
    k,
    0,
    0
  )

# C[M,N] = A[M,K] * B[N,K]^T.
def mm3_nt(
  fuel: Nat,
  dr: Array<F32> & Array<F32> & F32,
  c: Array<F32>,
  +m: U32,
  +n: U32,
  +k: U32,
  +row: U32,
  +col: U32
) -> T3:
  match fuel:
    case 0n:
      match dr:
        case (a1, b1, acc):
          T3{c, a1, b1}
    case 1n+q:
      match dr:
        case (a1, b1, acc):
          p = U32.add(U32.mul(row, n), col)
          c1 = Array.set(F32, c, p, acc)
          +col1 = U32.add(col, 1)
          +wrap = U32.is_eq(col1, n)
          +row1 = U32.add(row, Bool.to_u32(wrap))
          +col2 = U32.sub(col1, U32.mul(n, Bool.to_u32(wrap)))
          +ai = U32.mul(row1, k)
          +bi = U32.mul(col2, k)
          mm3_nt(
            q,
            dot(
              U32.to_nat(k),
              Array.get(F32, a1, ai),
              Array.get(F32, b1, bi),
              ai,
              bi,
              1,
              1,
              0.0
            ),
            c1,
            m,
            n,
            k,
            row1,
            col2
          )

def matmul3_nt(a: Array<F32>, b: Array<F32>, c: Array<F32>, +m: U32, +n: U32, +k: U32) -> T3:
  mm3_nt(
    U32.to_nat(U32.mul(m, n)),
    dot(
      U32.to_nat(k),
      Array.get(F32, a, 0),
      Array.get(F32, b, 0),
      0,
      0,
      1,
      1,
      0.0
    ),
    c,
    m,
    n,
    k,
    0,
    0
  )
