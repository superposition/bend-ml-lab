# Bend 2 notes from building this

These are the rules that cost time while writing `src/tensor.bend`,
`src/mlp.bend`, and the benchmarks. All verified against Bend **2.0.40**.
If a future release changes one, this file is stale by definition.

## 1. `match` only scrutinizes a parameter or a field

Rejected:

```python
match Array.get(F32, a, i):   # computed value
  case (a1, x): ...
```

Accepted:

```python
def get(fuel: Nat, ar: Array<F32> & F32, i: U32) -> ...:
  match ar:                    # ar is a parameter
    case (a1, x): ...
```

The accepted form is why every array-reading loop in `tensor.bend` takes the
pair as a parameter, and why callers pass `Array.get(...)` as an argument
instead of destructuring it. The compiler's message is explicit:

> a parameter or field scrutinee (a match cannot scrutinize a computed value:
> give it its own def)

Consequence: **a function that returns a tuple is consumed by another def**, not
destructured at the call site. `go.end(dot(...))` exists for this reason.

## 2. Match order follows binder order

```python
def f(fuel: Nat, ar: A, br: B) -> ...:
  match fuel:        # first parameter first
    case 0n: ...
    case 1n+p:
      match ar: ...  # then the second
        ...
```

Matching `br` before `fuel` is rejected with *"`fuel` can't be matched in this
position (it is matched after a match on a later binder)"*. Nested matches are
fine; out-of-order matches are not.

## 3. Loop parameters: put the shrinking one first

Termination is checked left to right: each argument before the decreasing one
must pass through unchanged.

```python
# rejected: w2 is rewritten before n shrinks
def scale.row(w2: List<F32>, c: Nat, n: Nat, s: F32): ...

# accepted: n first, everything after it is free
def scale.row(n: Nat, w2: List<F32>, c: Nat, s: F32): ...
```

For loops over an array, `fuel: Nat` is the first parameter; the array may be
rewritten in the recursive call.

## 4. A pattern field used twice needs `+`

```python
case 1n+p:                 # p affine, one use
case 1n++p:                # p reusable
```

The same rule applies to function parameters and record fields: `+x: T` and
`case K{+x}`. It requires `T: Data`. `Array<T>` is a `Type`, so arrays can
**never** be marked `+`; clone or restructure instead.

## 5. `Type` vs `Data` decides what can be copied

- `U32`, `F32`, `Nat`, `List<A>` with `A: Data`, and records of those are
  `Data`: copyable with `+`.
- `Array<T>` is `Type`: exactly one owner, reads hand it back, writes rebind it.
- A `List<F32>` written as `List<F32>` is affine (`&1`); the copyable type is
  `List<&2, F32>`. This matters for record fields: `type Model is Data:
  M{xs: List<F32>}` is rejected; use `List<&2, F32>`.

## 6. Do-block lets are annotated; binds need `do`

```python
def main() -> IO(Unit):
  do IO<Unit>:
    a : Array<F32> = Array.new(F32, 6n, 0.0)
    a : Array<F32> = Array.set(F32, a, 0, 1.0)
    x : F32 <- IO.now()
```

Inside a `do` block every let needs `x : T = v`. A `<-` bind is only legal
inside a `do` block; a `def` that returns `IO` but uses `match` needs a
`do IO<...>:` inside the arm that binds.

## 7. No negative literals

`-0.8` is a parse error. Use `F32.neg(0.8)` or `F32.sub(0.0, 0.8)`.

## 8. Operators need a type annotation

Inside `(.. : T)`, operators desugar to `T.add` etc. A bare `1n + i` is an
error:

```python
lget(w1, (1n + i : Nat))
```

## 9. Array shapes are powers of two

`Array.new(T, d, v)` allocates `2^d` slots. `[v : T*n]` repeats one value; it is
not a list literal. Indexes wrap. For an `M x N` tensor with arbitrary `M,N`,
allocate at `U32.log2(M*N)` rounded up and treat the logical shape separately.

## 10. Parallel array work is `@unsafe`

`Array.fork(T, a) -> Array<T> & Array<T>` aliases one array into two lanes;
`Array.join(T, a, b)` returns `a` after both lanes are done. The safety
obligation is yours: every lane must write a disjoint band and reads must be
read-only. `src/parallel.bend::tree` does this by splitting output rows.

`@unsafe` is also the only way to share an array across a parallel call. It
makes `bend --check-only` report `SOME PROOFS FAIL` for the defs that use it;
that is expected, not a bug. That is why the fork/join tree lives in its own
file, `src/parallel.bend`: the safe core (`tensor.bend`, `mlp.bend`) still
reports `ALL PROOFS CHECK`.

## 11. Parallel calls are writes, not threads

```python
l r = left(...) right(...)
Array.join(F32, l, r)
```

Both sides must do roughly equal work. Depth 2 on a 1024x1024 matmul is *slower*
than the sequential version because four lanes each get a quarter of a 4-second
job and the fork/join overhead is not free.

## 12. Targets and tooling

- `bend file.bend` checks and runs `main`; there is no `bend run` subcommand.
- `-o out` needs clang 14+; `!` needs clang 19+.
- `!` is a no-op on the JS target: `bend file.bend` ignores it.
- The JS target is one core and ~7x slower on the 1024 matmul here.
- Native build time is ~1.5-2 s per program (one C file, clang, no incremental
  builds).
- GPU: CUDA 12/13 or Metal, and **concurrent managed access**. WSL2 lacks it:
  `bend: CUDA device lacks concurrent managed access (WSL2 lacks it)`.
- No native Windows. No `U64`, `I64`, or `F64`; floats are `F32` only.
- Module constructors are qualified: `M.M{...}`, even with `import x as M`.

## 13. The checker is part of the cost model

`bend --check-only` runs the type checker, which is also the proof checker. It is
fast enough to run after every edit, but it does not check the loops for
correctness — only types, quantities, and termination. The benchmarks here are
not proved; `LAWS.bend` is still empty.
