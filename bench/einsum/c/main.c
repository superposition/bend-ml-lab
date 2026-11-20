/* C twin for bench/einsum/main.bend.
 *
 *   gcc -O3 -o mm_seq main.c
 *   gcc -O3 -fopenmp -o mm_par main.c -DPAR
 *   ./mm_seq 1024
 *
 * Same xorshift32 generator, same row-major float32 loops, same checksum
 * order as the Bend program, so the checksums are comparable.
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>

static inline uint32_t prng(uint32_t x) {
  x ^= x << 13;
  x ^= x >> 17;
  x ^= x << 5;
  return x;
}

static void gen(float *a, size_t n, uint32_t seed) {
  for (size_t i = 0; i < n; i++) {
    seed = prng(seed);
    a[i] = (float)(seed >> 8) / 16777216.0f;
  }
}

static float checksum(const float *a, size_t n) {
  float s = 0.0f;
  for (size_t i = 0; i < n; i++) s += a[i];
  return s;
}

static double now_ms(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
}

#ifdef PAR
#define MODE "par"
#else
#define MODE "seq"
#endif

int main(int argc, char **argv) {
  size_t n = (argc > 1) ? (size_t)strtoul(argv[1], NULL, 10) : 1024;
  size_t cells = n * n;
  float *a = malloc(cells * sizeof(float));
  float *b = malloc(cells * sizeof(float));
  float *c = malloc(cells * sizeof(float));
  if (!a || !b || !c) return 1;

  double t0 = now_ms();
  gen(a, cells, 7);
  gen(b, cells, 11);
  double t1 = now_ms();

  double t2 = now_ms();
#ifdef PAR
#pragma omp parallel for schedule(static)
#endif
  for (size_t i = 0; i < n; i++) {
    for (size_t j = 0; j < n; j++) {
      float s = 0.0f;
      for (size_t k = 0; k < n; k++) {
        s += a[i * n + k] * b[k * n + j];
      }
      c[i * n + j] = s;
    }
  }
  double t3 = now_ms();

  printf("C,%s,gen,%f,%f\n", MODE, (double)checksum(a, cells) + (double)checksum(b, cells), t1 - t0);
  printf("C,%s,matmul,%f,%f\n", MODE, (double)checksum(c, cells), t3 - t2);
  free(a); free(b); free(c);
  return 0;
}
