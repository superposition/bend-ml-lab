import Base
import ./tensor.bend as T

# parallel.bend — the @unsafe half of the tensor library.
#
# `Array.fork` hands two aliases of one array to two lanes, so this file is
# deliberately separate from tensor.bend: importing it makes `bend
# --check-only` report the unsafe defs (tree, pmatmul). The training and WSP
# code does not import it.
#
# Safety obligation: every lane writes a disjoint band of output rows and
# only reads A and B. `tree` splits C by rows; `Array.join` merges handles.

# ---------------------------------------------------------------------------
# Parallel matmul (CPU fork/join; `!` sends the same tree to the GPU)
# ---------------------------------------------------------------------------
#
# `Array.fork` is @unsafe: it hands two aliases of one array to two lanes.
# Safety here is structural — every lane writes a disjoint band of C rows
# and only reads A and B. `Array.join` merges the handles afterwards.
#
# depth d => 2^d lanes; the row band per lane is m / 2^d.

def tree(
  depth: Nat,
  fa: Array<F32> & Array<F32>,
  fb: Array<F32> & Array<F32>,
  fc: Array<F32> & Array<F32>,
  +row0: U32,
  +rows: U32,
  +m: U32,
  +n: U32,
  +k: U32
) -> Array<F32>:
  match depth:
    case 0n:
      match fa:
        case (a1, a2):
          match fb:
            case (b1, b2):
              match fc:
                case (c1, c2):
                  +ai = U32.mul(row0, k)
                  r = T.mm(
                    U32.to_nat(U32.mul(rows, n)),
                    T.dot(
                      U32.to_nat(k),
                      Array.get(F32, a1, ai),
                      Array.get(F32, b1, 0),
                      ai,
                      0,
                      1,
                      n,
                      0.0
                    ),
                    c1,
                    m,
                    n,
                    k,
                    row0,
                    0
                  )
                  Array.join(F32, r, c2)
    case 1n++p:
      match fa:
        case (a1, a2):
          match fb:
            case (b1, b2):
              match fc:
                case (c1, c2):
                  +half = U32.shrn(rows, 1n)
                  l r = tree(
                      p,
                      Array.fork(F32, a1),
                      Array.fork(F32, b1),
                      Array.fork(F32, c1),
                      row0,
                      half,
                      m,
                      n,
                      k
                    )
                    tree(
                      p,
                      Array.fork(F32, a2),
                      Array.fork(F32, b2),
                      Array.fork(F32, c2),
                      U32.add(row0, half),
                      half,
                      m,
                      n,
                      k
                    )
                  Array.join(F32, l, r)

# pmatmul: 2^depth lanes. depth 0 degenerates to the sequential matmul.
def pmatmul(depth: Nat, a: Array<F32>, b: Array<F32>, c: Array<F32>, +m: U32, +n: U32, +k: U32) -> Array<F32>:
  match depth:
    case 0n:
      T.matmul(a, b, c, m, n, k)
    case 1n++p:
      tree(
        p,
        Array.fork(F32, a),
        Array.fork(F32, b),
        Array.fork(F32, c),
        0,
        m,
        m,
        n,
        k
      )


# ---------------------------------------------------------------------------
# Transposed-B variants for the GPU
# ---------------------------------------------------------------------------

def tree_bt(
  depth: Nat,
  fa: Array<F32> & Array<F32>,
  fb: Array<F32> & Array<F32>,
  fc: Array<F32> & Array<F32>,
  +row0: U32,
  +rows: U32,
  +m: U32,
  +n: U32,
  +k: U32
) -> Array<F32>:
  match depth:
    case 0n:
      match fa:
        case (a1, a2):
          match fb:
            case (b1, b2):
              match fc:
                case (c1, c2):
                  +ai = U32.mul(row0, k)
                  r = T.mm_bt(
                    U32.to_nat(U32.mul(rows, n)),
                    T.dot(
                      U32.to_nat(k),
                      Array.get(F32, a1, ai),
                      Array.get(F32, b1, 0),
                      ai,
                      0,
                      1,
                      1,
                      0.0
                    ),
                    c1,
                    m,
                    n,
                    k,
                    row0,
                    0
                  )
                  Array.join(F32, r, c2)
    case 1n++p:
      match fa:
        case (a1, a2):
          match fb:
            case (b1, b2):
              match fc:
                case (c1, c2):
                  +half = U32.shrn(rows, 1n)
                  l r = tree_bt(
                      p,
                      Array.fork(F32, a1),
                      Array.fork(F32, b1),
                      Array.fork(F32, c1),
                      row0,
                      half,
                      m,
                      n,
                      k
                    )
                    tree_bt(
                      p,
                      Array.fork(F32, a2),
                      Array.fork(F32, b2),
                      Array.fork(F32, c2),
                      U32.add(row0, half),
                      half,
                      m,
                      n,
                      k
                    )
                  Array.join(F32, l, r)

def pmatmul_bt(depth: Nat, a: Array<F32>, bt: Array<F32>, c: Array<F32>, +m: U32, +n: U32, +k: U32) -> Array<F32>:
  match depth:
    case 0n:
      T.matmul_bt(a, bt, c, m, n, k)
    case 1n++p:
      tree_bt(
        p,
        Array.fork(F32, a),
        Array.fork(F32, bt),
        Array.fork(F32, c),
        0,
        m,
        m,
        n,
        k
      )

# ---------------------------------------------------------------------------
# 2D block tree: rd row levels, then cd column levels.
# Leaves are (rows / 2^rd) x (cols / 2^cd) output blocks, all in parallel.
# ---------------------------------------------------------------------------

def tree2(
  rd: Nat,
  +cd: Nat,
  fa: Array<F32> & Array<F32>,
  fb: Array<F32> & Array<F32>,
  fc: Array<F32> & Array<F32>,
  +row0: U32,
  +rows: U32,
  +col0: U32,
  +cols: U32,
  +m: U32,
  +n: U32,
  +k: U32
) -> Array<F32>:
  match rd:
    case 1n++p:
      match fa:
        case (a1, a2):
          match fb:
            case (b1, b2):
              match fc:
                case (c1, c2):
                  +half = U32.shrn(rows, 1n)
                  l r = tree2(
                      p,
                      cd,
                      Array.fork(F32, a1),
                      Array.fork(F32, b1),
                      Array.fork(F32, c1),
                      row0,
                      half,
                      col0,
                      cols,
                      m,
                      n,
                      k
                    )
                    tree2(
                      p,
                      cd,
                      Array.fork(F32, a2),
                      Array.fork(F32, b2),
                      Array.fork(F32, c2),
                      U32.add(row0, half),
                      half,
                      col0,
                      cols,
                      m,
                      n,
                      k
                    )
                  Array.join(F32, l, r)
    case 0n:
      match cd:
        case 1n++q:
          match fa:
            case (a1, a2):
              match fb:
                case (b1, b2):
                  match fc:
                    case (c1, c2):
                      +half = U32.shrn(cols, 1n)
                      l r = tree2(
                          0n,
                          q,
                          Array.fork(F32, a1),
                          Array.fork(F32, b1),
                          Array.fork(F32, c1),
                          row0,
                          rows,
                          col0,
                          half,
                          m,
                          n,
                          k
                        )
                        tree2(
                          0n,
                          q,
                          Array.fork(F32, a2),
                          Array.fork(F32, b2),
                          Array.fork(F32, c2),
                          row0,
                          rows,
                          U32.add(col0, half),
                          half,
                          m,
                          n,
                          k
                        )
                      Array.join(F32, l, r)
        case 0n:
          match fa:
            case (a1, a2):
              match fb:
                case (b1, b2):
                  match fc:
                    case (c1, c2):
                      r = T.matmul_bt_range(rows, cols, a1, b1, c1, n, k, row0, col0)
                      Array.join(F32, r, c2)

def pmatmul2_bt(
  rd: Nat,
  +cd: Nat,
  a: Array<F32>,
  bt: Array<F32>,
  c: Array<F32>,
  +m: U32,
  +n: U32,
  +k: U32
) -> Array<F32>:
  tree2(
    rd,
    cd,
    Array.fork(F32, a),
    Array.fork(F32, bt),
    Array.fork(F32, c),
    0,
    m,
    0,
    n,
    m,
    n,
    k
  )


# Tiled leaf: two rows per lane, 2x2 tiles.
def leaf2.go(
  pa: Array<F32> & Array<F32>,
  pb: Array<F32> & Array<F32>,
  c: Array<F32>,
  +row0: U32,
  +cols: U32,
  +col0: U32,
  +n: U32,
  +k: Nat
) -> Array<F32>:
  match pa:
    case (a0, a1):
      match pb:
        case (b0, b1):
          T.tile.done(T.tiles.start(cols, a0, a1, b0, b1, c, row0, col0, n, k))

def leaf2(
  a: Array<F32>,
  bt: Array<F32>,
  c: Array<F32>,
  +row0: U32,
  +cols: U32,
  +col0: U32,
  +n: U32,
  +k: Nat
) -> Array<F32>:
  leaf2.go(
    Array.fork(F32, a),
    Array.fork(F32, bt),
    c,
    row0,
    cols,
    col0,
    n,
    k
  )

def tree3(
  rd: Nat,
  +cd: Nat,
  fa: Array<F32> & Array<F32>,
  fb: Array<F32> & Array<F32>,
  fc: Array<F32> & Array<F32>,
  +row0: U32,
  +rows: U32,
  +col0: U32,
  +cols: U32,
  +m: U32,
  +n: U32,
  +k: U32
) -> Array<F32>:
  match rd:
    case 1n++p:
      match fa:
        case (a1, a2):
          match fb:
            case (b1, b2):
              match fc:
                case (c1, c2):
                  +half = U32.shrn(rows, 1n)
                  l r = tree3(
                      p,
                      cd,
                      Array.fork(F32, a1),
                      Array.fork(F32, b1),
                      Array.fork(F32, c1),
                      row0,
                      half,
                      col0,
                      cols,
                      m,
                      n,
                      k
                    )
                    tree3(
                      p,
                      cd,
                      Array.fork(F32, a2),
                      Array.fork(F32, b2),
                      Array.fork(F32, c2),
                      U32.add(row0, half),
                      half,
                      col0,
                      cols,
                      m,
                      n,
                      k
                    )
                  Array.join(F32, l, r)
    case 0n:
      match cd:
        case 1n++q:
          match fa:
            case (a1, a2):
              match fb:
                case (b1, b2):
                  match fc:
                    case (c1, c2):
                      +half = U32.shrn(cols, 1n)
                      l r = tree3(
                          0n,
                          q,
                          Array.fork(F32, a1),
                          Array.fork(F32, b1),
                          Array.fork(F32, c1),
                          row0,
                          rows,
                          col0,
                          half,
                          m,
                          n,
                          k
                        )
                        tree3(
                          0n,
                          q,
                          Array.fork(F32, a2),
                          Array.fork(F32, b2),
                          Array.fork(F32, c2),
                          row0,
                          rows,
                          U32.add(col0, half),
                          half,
                          m,
                          n,
                          k
                        )
                      Array.join(F32, l, r)
        case 0n:
          match fa:
            case (a1, a2):
              match fb:
                case (b1, b2):
                  match fc:
                    case (c1, c2):
                      r = leaf2(a1, b1, c1, row0, cols, col0, n, U32.to_nat(k))
                      Array.join(F32, r, c2)

def pmatmul3_bt(
  rd: Nat,
  cd: Nat,
  a: Array<F32>,
  bt: Array<F32>,
  c: Array<F32>,
  +m: U32,
  +n: U32,
  +k: U32
) -> Array<F32>:
  tree3(
    rd,
    cd,
    Array.fork(F32, a),
    Array.fork(F32, bt),
    Array.fork(F32, c),
    0,
    m,
    0,
    n,
    m,
    n,
    k
  )

# 4x4 tiled leaf: four rows per lane, 4x4 output tiles.
def leaf4.b(pa01: Array<F32> & Array<F32>, pa23: Array<F32> & Array<F32>,
            pb01: Array<F32> & Array<F32>, pb23: Array<F32> & Array<F32>,
            c: Array<F32>, +row0: U32, +cols: U32, +col0: U32, +n: U32, +k: Nat) -> Array<F32>:
  match pa01:
    case (a0, a1):
      match pa23:
        case (a2, a3):
          match pb01:
            case (b0, b1):
              match pb23:
                case (b2, b3):
                  T.tile4.done(T.tiles4.start(cols, a0, a1, a2, a3, b0, b1, b2, b3, c, row0, col0, n, k))

def leaf4.a(pa: Array<F32> & Array<F32>, pb: Array<F32> & Array<F32>,
            c: Array<F32>, +row0: U32, +cols: U32, +col0: U32, +n: U32, +k: Nat) -> Array<F32>:
  match pa:
    case (a01, a23):
      match pb:
        case (b01, b23):
          leaf4.b(
            Array.fork(F32, a01), Array.fork(F32, a23),
            Array.fork(F32, b01), Array.fork(F32, b23),
            c, row0, cols, col0, n, k)

def leaf4(a: Array<F32>, bt: Array<F32>, c: Array<F32>,
          +row0: U32, +cols: U32, +col0: U32, +n: U32, +k: Nat) -> Array<F32>:
  leaf4.a(Array.fork(F32, a), Array.fork(F32, bt), c, row0, cols, col0, n, k)

def tree4(rd: Nat, +cd: Nat, fa: Array<F32> & Array<F32>, fb: Array<F32> & Array<F32>,
          fc: Array<F32> & Array<F32>, +row0: U32, +rows: U32, +col0: U32, +cols: U32,
          +m: U32, +n: U32, +k: U32) -> Array<F32>:
  match rd:
    case 1n++p:
      match fa:
        case (a1, a2):
          match fb:
            case (b1, b2):
              match fc:
                case (c1, c2):
                  +half = U32.shrn(rows, 1n)
                  l r = tree4(p, cd, Array.fork(F32, a1), Array.fork(F32, b1), Array.fork(F32, c1),
                      row0, half, col0, cols, m, n, k)
                    tree4(p, cd, Array.fork(F32, a2), Array.fork(F32, b2), Array.fork(F32, c2),
                      U32.add(row0, half), half, col0, cols, m, n, k)
                  Array.join(F32, l, r)
    case 0n:
      match cd:
        case 1n++q:
          match fa:
            case (a1, a2):
              match fb:
                case (b1, b2):
                  match fc:
                    case (c1, c2):
                      +half = U32.shrn(cols, 1n)
                      l r = tree4(0n, q, Array.fork(F32, a1), Array.fork(F32, b1), Array.fork(F32, c1),
                          row0, rows, col0, half, m, n, k)
                        tree4(0n, q, Array.fork(F32, a2), Array.fork(F32, b2), Array.fork(F32, c2),
                          row0, rows, U32.add(col0, half), half, m, n, k)
                      Array.join(F32, l, r)
        case 0n:
          match fa:
            case (a1, a2):
              match fb:
                case (b1, b2):
                  match fc:
                    case (c1, c2):
                      r = leaf4(a1, b1, c1, row0, cols, col0, n, U32.to_nat(k))
                      Array.join(F32, r, c2)

def pmatmul4_bt(rd: Nat, cd: Nat, a: Array<F32>, bt: Array<F32>, c: Array<F32>,
                +m: U32, +n: U32, +k: U32) -> Array<F32>:
  tree4(rd, cd, Array.fork(F32, a), Array.fork(F32, bt), Array.fork(F32, c),
    0, m, 0, n, m, n, k)

# Indexed fold with Array.get_unchecked (no size/bounds work).
def csumu.go(fuel: Nat, ar: Array<F32> & F32, +i: U32, acc: F32) -> F32:
  match fuel:
    case 0n:
      acc
    case 1n+p:
      match ar:
        case (a1, x):
          csumu.go(p, Array.get_unchecked(F32, a1, U32.add(i, 1)), U32.add(i, 1), F32.add(acc, x))

def csumu(cells: U32, a: Array<F32>) -> F32:
  csumu.go(U32.to_nat(cells), Array.get_unchecked(F32, a, 0), 0, 0.0)
