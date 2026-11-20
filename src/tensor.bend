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
          relu.go(p, Array.get(F32, a1, U32.add(i, 1)), U32.add(i, 1))

def relu(cells: U32, a: Array<F32>) -> Array<F32>:
  relu.go(U32.to_nat(cells), Array.get(F32, a, 0), 0)

def map_scale.go(fuel: Nat, ar: Array<F32> & F32, +i: U32, s: F32) -> Array<F32>:
  match fuel:
    case 0n:
      match ar:
        case (a1, x):
          a1
    case 1n++p:
      match ar:
        case (a1, x):
          map_scale.go(p, Array.get(F32, a1, U32.add(i, 1)), U32.add(i, 1), s)

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
