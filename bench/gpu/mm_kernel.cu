#include <cuda_runtime.h>
#define T 16
extern "C" __global__ void mm_kernel(const float* A, const float* B, float* C, int n) {
  __shared__ float As[T][T];
  __shared__ float Bs[T][T];
  int tx = threadIdx.x, ty = threadIdx.y;
  int row = blockIdx.y * T + ty;
  int col = blockIdx.x * T + tx;
  if (row >= n || col >= n) return;
  float acc = 0.0f;
  for (int t = 0; t < n / T; t++) {
    As[ty][tx] = A[row * n + t * T + tx];
    Bs[ty][tx] = B[(t * T + ty) * n + col];
    __syncthreads();
    for (int k = 0; k < T; k++) acc += As[ty][k] * Bs[k][tx];
    __syncthreads();
  }
  C[row * n + col] = acc;
}
