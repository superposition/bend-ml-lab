"""NumPy reference for the same generator and matmul.

NumPy uses BLAS for `a @ b`, so this is a reference for the result, not
an apples-to-apples naive-loop comparison. Run: python main.py 1024
"""
import sys
import time
import numpy as np

n = int(sys.argv[1]) if len(sys.argv) > 1 else 1024
cells = n * n

def prng(x):
    x &= 0xFFFFFFFF
    x ^= (x << 13) & 0xFFFFFFFF
    x ^= x >> 17
    x ^= (x << 5) & 0xFFFFFFFF
    return x & 0xFFFFFFFF

def gen(nvals, seed):
    out = np.empty(nvals, dtype=np.float32)
    for i in range(nvals):
        seed = prng(seed)
        out[i] = np.float32((seed >> 8) / 16777216.0)
    return out

t0 = time.perf_counter()
a = gen(cells, 7)
b = gen(cells, 11)
t1 = time.perf_counter()
A = a.reshape(n, n)
B = b.reshape(n, n)
c = (A @ B).astype(np.float32)
t2 = time.perf_counter()

print(f"NumPy,blas,gen,{np.float32(a.sum() + b.sum())},{(t1-t0)*1000:.3f}")
print(f"NumPy,blas,matmul,{np.float32(c.sum())},{(t2-t1)*1000:.3f}")
