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

