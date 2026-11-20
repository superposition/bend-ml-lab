import sys
ptx_path, out_path = sys.argv[1], sys.argv[2]
ptx = open(ptx_path).read()
esc = ptx.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
c = r'''#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Device-buffer matmul as a Bend C effect.
 *
 * The Bend Array<F32> is a contiguous block of u64 cells (TAG_ARR), each cell
 * holding the raw f32 bits (f32_unbox / f32_rewrap). This effect copies the
 * operands to device memory, runs the tiled kernel, and writes the result back
 * into a copy of the output array. No managed heap, no page migration.
 *
 * Native only: the .js twin throws.
 */

typedef int CUresult;
typedef int CUdevice;
typedef unsigned long long CUdeviceptr;
typedef void* CUcontext;
typedef void* CUmodule;
typedef void* CUfunction;

static const char MM_PTX[] =
"''' + esc + r'''";

static void* cuda_lib;
static CUresult (*p_cuInit)(unsigned);
static CUresult (*p_cuDeviceGet)(CUdevice*, int);
static CUresult (*p_cuCtxCreate)(CUcontext*, unsigned, CUdevice);
static CUresult (*p_cuCtxSetCurrent)(CUcontext);
static CUresult (*p_cuModuleLoadDataEx)(CUmodule*, const void*, unsigned, void*, void**);
static CUresult (*p_cuModuleGetFunction)(CUfunction*, CUmodule, const char*);
static CUresult (*p_cuMemAlloc)(CUdeviceptr*, size_t);
static CUresult (*p_cuMemFree)(CUdeviceptr);
static CUresult (*p_cuMemcpyHtoD)(CUdeviceptr, const void*, size_t);
static CUresult (*p_cuMemcpyDtoH)(void*, CUdeviceptr, size_t);
static CUresult (*p_cuLaunchKernel)(CUfunction, unsigned, unsigned, unsigned,
                                    unsigned, unsigned, unsigned, unsigned,
                                    void*, void**, void**);
static CUresult (*p_cuCtxSynchronize)(void);

static void* cuda_sym(const char* name, const char* alt) {
  void* p = dlsym(cuda_lib, name);
  if (!p && alt) p = dlsym(cuda_lib, alt);
  return p;
}

static bool cuda_load(void) {
  if (cuda_lib) return true;
  cuda_lib = dlopen("libcuda.so.1", RTLD_NOW | RTLD_GLOBAL);
  if (!cuda_lib) cuda_lib = dlopen("libcuda.so", RTLD_NOW | RTLD_GLOBAL);
  if (!cuda_lib) return false;
  p_cuInit = cuda_sym("cuInit", NULL);
  p_cuDeviceGet = cuda_sym("cuDeviceGet", NULL);
  p_cuCtxCreate = cuda_sym("cuCtxCreate_v2", "cuCtxCreate");
  p_cuCtxSetCurrent = cuda_sym("cuCtxSetCurrent", NULL);
  p_cuModuleLoadDataEx = cuda_sym("cuModuleLoadDataEx", NULL);
  p_cuModuleGetFunction = cuda_sym("cuModuleGetFunction", NULL);
  p_cuMemAlloc = cuda_sym("cuMemAlloc_v2", "cuMemAlloc");
  p_cuMemFree = cuda_sym("cuMemFree_v2", "cuMemFree");
  p_cuMemcpyHtoD = cuda_sym("cuMemcpyHtoD_v2", "cuMemcpyHtoD");
  p_cuMemcpyDtoH = cuda_sym("cuMemcpyDtoH_v2", "cuMemcpyDtoH");
  p_cuLaunchKernel = cuda_sym("cuLaunchKernel", NULL);
  p_cuCtxSynchronize = cuda_sym("cuCtxSynchronize", NULL);
  return p_cuInit && p_cuDeviceGet && p_cuCtxCreate && p_cuCtxSetCurrent &&
         p_cuModuleLoadDataEx && p_cuModuleGetFunction && p_cuMemAlloc &&
         p_cuMemFree && p_cuMemcpyHtoD && p_cuMemcpyDtoH &&
         p_cuLaunchKernel && p_cuCtxSynchronize;
}

static CUfunction mm_fn;

static bool cuda_setup(void) {
  if (mm_fn) return true;
  if (!cuda_load()) return false;
  if (p_cuInit(0) != 0) return false;
  CUdevice dev;
  if (p_cuDeviceGet(&dev, 0) != 0) return false;
  CUcontext ctx;
  if (p_cuCtxCreate(&ctx, 0, dev) != 0) return false;
  if (p_cuCtxSetCurrent(ctx) != 0) return false;
  CUmodule mod;
  if (p_cuModuleLoadDataEx(&mod, MM_PTX, 0, NULL, NULL) != 0) return false;
  if (p_cuModuleGetFunction(&mm_fn, mod, "mm_kernel") != 0) return false;
  return true;
}

Term device_mm_run(Env e, Term* f, IoWork* w) {
  if (!cuda_setup()) err_fail("device_mm: no CUDA driver");
  unsigned m = (unsigned)f[3], n = (unsigned)f[4], k = (unsigned)f[5];
  if (m == 0 || n == 0 || k == 0 || (n & 15) != 0) {
    err_fail("device_mm: n must be a non-zero multiple of 16");
  }
  unsigned ca = 1u << blk_cls(f[0]), cb = 1u << blk_cls(f[1]), cc = 1u << blk_cls(f[2]);
  /* each u64 block word holds two u32 elements */
  if (m * k > 2 * ca || k * n > 2 * cb || m * n > 2 * cc) {
    err_fail("device_mm: array too small");
  }
  size_t ab = (size_t)m * k * 4, bb = (size_t)k * n * 4, cbts = (size_t)m * n * 4;
  u64 la = blk_loc(e.mem, f[0]), lb = blk_loc(e.mem, f[1]);
  u32* ea = (u32*)(e.mem + la);
  u32* eb = (u32*)(e.mem + lb);

  static CUdeviceptr dA, dB, dC;
  static size_t capA, capB, capC;
  if (ab > capA) {
    if (dA) p_cuMemFree(dA);
    if (p_cuMemAlloc(&dA, ab)) err_fail("device_mm: device allocation failed");
    capA = ab;
  }
  if (bb > capB) {
    if (dB) p_cuMemFree(dB);
    if (p_cuMemAlloc(&dB, bb)) err_fail("device_mm: device allocation failed");
    capB = bb;
  }
  if (cbts > capC) {
    if (dC) p_cuMemFree(dC);
    if (p_cuMemAlloc(&dC, cbts)) err_fail("device_mm: device allocation failed");
    capC = cbts;
  }
  /* the Array<F32> block is already f32 bytes: copy it straight to the device */
  if (p_cuMemcpyHtoD(dA, ea, ab) || p_cuMemcpyHtoD(dB, eb, bb)) {
    err_fail("device_mm: copy in failed");
  }
  int nn = (int)n;
  void* args[] = { &dA, &dB, &dC, &nn };
  if (p_cuLaunchKernel(mm_fn, n / 16, n / 16, 1, 16, 16, 1, 0, NULL, args, NULL)) {
    err_fail("device_mm: launch failed");
  }
  Term out = blk_copy(e, f[2]);
  u64 lc = blk_loc(e.mem, out);
  u32* ec = (u32*)(e.mem + lc);
  if (p_cuCtxSynchronize() || p_cuMemcpyDtoH(ec, dC, cbts)) {
    err_fail("device_mm: copy out failed");
  }
  return out;
}

static void __attribute__((constructor)) device_mm_use(void) {
  io_eff(CID(Device.mm), device_mm_run);
}
'''
open(out_path, "w").write(c)
print("wrote", out_path, len(c))
