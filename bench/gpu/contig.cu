// Contiguous-buffer ceiling for the 1024x1024 F32 matmul.
// Device buffers + explicit copies (the thing Bend's managed tree cannot do).
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define T 16

__global__ void mm(const float* A, const float* B, float* C, int n) {
  __shared__ float As[T][T];
  __shared__ float Bs[T][T];
  int tx = threadIdx.x, ty = threadIdx.y;
  int row = blockIdx.y * T + ty;
  int col = blockIdx.x * T + tx;
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

static inline unsigned prng(unsigned x){x^=x<<13;x^=x>>17;x^=x<<5;return x;}
static void gen(float* a, size_t n, unsigned s){for(size_t i=0;i<n;i++){s=prng(s);a[i]=(float)(s>>8)/16777216.0f;}}
static double ms(cudaEvent_t a, cudaEvent_t b){float v;cudaEventElapsedTime(&v,a,b);return v;}

int main() {
  int n = 1024; size_t cells = (size_t)n * n, bytes = cells * sizeof(float);
  float *hA = (float*)malloc(bytes), *hB = (float*)malloc(bytes), *hC = (float*)malloc(bytes);
  gen(hA, cells, 7); gen(hB, cells, 11);

  int cma = 0; cudaDeviceGetAttribute(&cma, cudaDevAttrConcurrentManagedAccess, 0);
  printf("cma=%d\n", cma);

  float *dA, *dB, *dC;
  cudaMalloc(&dA, bytes); cudaMalloc(&dB, bytes); cudaMalloc(&dC, bytes);
  cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);

  cudaEventRecord(a);
  cudaMemcpy(dA, hA, bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(dB, hB, bytes, cudaMemcpyHostToDevice);
  cudaEventRecord(b); cudaEventSynchronize(b);
  double copy_in = ms(a, b);

  dim3 grid(n / T, n / T), block(T, T);
  cudaEventRecord(a);
  mm<<<grid, block>>>(dA, dB, dC, n);
  cudaEventRecord(b); cudaEventSynchronize(b);
  double kernel = ms(a, b);

  cudaEventRecord(a);
  cudaMemcpy(hC, dC, bytes, cudaMemcpyDeviceToHost);
  cudaEventRecord(b); cudaEventSynchronize(b);
  double copy_out = ms(a, b);

  double sum = 0; for (size_t i = 0; i < cells; i++) sum += hC[i];
  printf("device,kernel_ms=%.3f,copy_in_ms=%.3f,copy_out_ms=%.3f,total_ms=%.3f,sum=%.1f\n",
    kernel, copy_in, copy_out, kernel + copy_in + copy_out, sum);

  // Managed variant: no explicit copies, same kernel.
  float *mA, *mB, *mC;
  cudaMallocManaged(&mA, bytes); cudaMallocManaged(&mB, bytes); cudaMallocManaged(&mC, bytes);
  memcpy(mA, hA, bytes); memcpy(mB, hB, bytes);
  cudaEventRecord(a);
  mm<<<grid, block>>>(mA, mB, mC, n);
  cudaEventRecord(b); cudaEventSynchronize(b);
  double mk = ms(a, b);
  double msum = 0; for (size_t i = 0; i < cells; i++) msum += mC[i];
  printf("managed,kernel_ms=%.3f,sum=%.1f\n", mk, msum);

  cudaFree(dA); cudaFree(dB); cudaFree(dC);
  cudaFree(mA); cudaFree(mB); cudaFree(mC);
  free(hA); free(hB); free(hC);
  return 0;
}
