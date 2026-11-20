import Base
import ../src/tensor.bend as T
import ./mlp_batch.bend as B

def show.f1(f: B.F1) -> IO(Unit):
  match f:
    case B.F1{h, x, w1, b1}:
      IO.print("h," ++ T.showA(h))

def show.d(d: B.D) -> IO(Unit):
  match d:
    case B.D{dh, w2, x, w1, b1, b2, loss, dw2, db2}:
      IO.print("dh," ++ T.showA(dh))

def show.ff(f: B.FF) -> IO(Unit):
  match f:
    case B.FF{db1, dw1, w1, w2, b1, b2, loss, dw2, db2}:
      do IO<Unit>:
        IO.print("db1," ++ T.showA(db1))
        IO.print("dw1," ++ T.showA(dw1))

def main() -> IO(Unit):
  do IO<Unit>:
    w1 : Array<F32> = T.map_scale(16, T.gen2(16, 7), 0.5)
    b1 : Array<F32> = Array.new(F32, 3n, 0.1)
    w2 : Array<F32> = T.map_scale(32, T.gen2(32, 9), 0.5)
    b2 : Array<F32> = Array.new(F32, 2n, 0.0)
    x : Array<F32> = B.mkx(64, 0)
    t : Array<F32> = B.mkt(64, 0)
    d : B.D = B.d(
      B.c(
        B.b(
          B.a(
            B.g(
              B.l(
                B.f2(
                  B.f1(T.matmul3(x, w1, Array.new(F32, U32.log2(512), 0.0), 64, 8, 2), b1, 64, 8),
                  w2, b2, 64, 4, 8),
                t, 64, 4),
             64, 4),
           64, 4, 8),
         64, 4, 8),
       64, 4),
     64, 8)
    w1h : Array<F32> = T.map_scale(16, T.gen2(16, 7), 0.5)
    b1h : Array<F32> = Array.new(F32, 3n, 0.1)
    xh : Array<F32> = B.mkx(64, 0)
    show.f1(B.f1(T.matmul3(xh, w1h, Array.new(F32, U32.log2(512), 0.0), 64, 8, 2), b1h, 64, 8))
    show.d(d)
    # recompute for ff (d was consumed by show.d)
    w1b : Array<F32> = T.map_scale(16, T.gen2(16, 7), 0.5)
    b1b : Array<F32> = Array.new(F32, 3n, 0.1)
    w2b : Array<F32> = T.map_scale(32, T.gen2(32, 9), 0.5)
    b2b : Array<F32> = Array.new(F32, 2n, 0.0)
    x2 : Array<F32> = B.mkx(64, 0)
    t2 : Array<F32> = B.mkt(64, 0)
    f : B.FF = B.ff(
      B.e(
        B.d(
          B.c(
            B.b(
              B.a(
                B.g(
                  B.l(
                    B.f2(
                      B.f1(T.matmul3(x2, w1b, Array.new(F32, U32.log2(512), 0.0), 64, 8, 2), b1b, 64, 8),
                      w2b, b2b, 64, 4, 8),
                    t2, 64, 4),
                 64, 4),
               64, 4, 8),
             64, 4, 8),
           64, 4),
         64, 8),
       64, 2, 8),
      64, 8)
    show.ff(f)
