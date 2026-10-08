/* Private terminal lowering for explicit ordered BF16 projections. Included
 * by ordered_executor.c so every existing native/sanitizer build covers it.
 * No symbol interception, request-time lookup, or global mutable handle. */
#include <limits.h>

typedef struct lf_ordered_projection_state {
  void *library;
  void *handle;
  CUdeviceptr workspace;
  CUdeviceptr scratch;
  int32_t (*destroy)(void *);
  int32_t (*set_stream)(void *, CUstream);
  int32_t (*set_workspace)(void *, void *, size_t);
  int32_t (*gemm)(void *, int, int, int, int, int, const void *,
    const void *, int, int, const void *, int, int, const void *,
    void *, int, int, int, int);
} lf_ordered_projection_state;

#define LF_ORDERED_PROJECTION_WORKSPACE (32U * 1024U * 1024U)

static int32_t lf_ordered_projection_close(lf_ordered_executor *e) {
  lf_ordered_projection_state *s = e->projections;
  if (s == NULL) return LF_OK;
  /* Graphs are destroyed first. Keep every failed resource for retry. */
  if (s->handle != NULL) {
    if (s->destroy(s->handle) != 0) return LF_DRIVER_FAILURE;
    s->handle = NULL;
  }
  if (s->scratch != 0) {
    int32_t status = lf_cuda_map_result(e->context->api->cuMemFree(s->scratch));
    if (status != LF_OK) return status;
    s->scratch = 0;
  }
  if (s->workspace != 0) {
    int32_t status = lf_cuda_map_result(e->context->api->cuMemFree(s->workspace));
    if (status != LF_OK) return status;
    s->workspace = 0;
  }
#if defined(__linux__)
  if (s->library != NULL && dlclose(s->library) != 0) return LF_DRIVER_FAILURE;
#endif
  free(s);
  e->projections = NULL;
  return LF_OK;
}

static int32_t lf_ordered_projection_run(
  lf_ordered_executor *e, const int32_t *p,
  CUdeviceptr x, CUdeviceptr w, CUdeviceptr y
) {
  const float alpha = 1.0f, beta = 0.0f;
  lf_ordered_projection_state *s = e->projections;
  /* Row-major X[M,K] times W[N,K]^T via column-major N,M,K.
   * CUDA_R_16BF=14, CUBLAS_COMPUTE_32F=68, tensor-op default=99.
   * Reduced-precision reductions are explicitly disabled at preparation. */
  return s->gemm(s->handle, 1, 0, p[1], p[0], p[2], &alpha,
    (const void *)(uintptr_t)w, 14, p[2],
    (const void *)(uintptr_t)x, 14, p[2], &beta,
    (void *)(uintptr_t)y, 14, p[1], 68, 99) == 0
    ? LF_OK : LF_DRIVER_FAILURE;
}

static int32_t lf_ordered_projection_validate(
  lf_ordered_kernel *kernel, const int32_t *p, const int64_t *sizes
) {
  if (p[0] == 0) {
    for (int i = 1; i < 6; ++i) if (p[i] != 0) return LF_INVALID_ARGUMENT;
    return LF_OK;
  }
  if (p[0] < 0 || p[1] <= 0 || p[2] <= 0) return LF_INVALID_ARGUMENT;
  uint64_t bytes[3] = {
    (uint64_t)p[0] * (uint64_t)p[2] * 2,
    (uint64_t)p[1] * (uint64_t)p[2] * 2,
    (uint64_t)p[0] * (uint64_t)p[1] * 2
  };
  for (int i = 0; i < 3; ++i) {
    int index = p[3+i];
    if (index < 0 || index >= kernel->argument_count ||
        bytes[i] > (uint64_t)sizes[index] || bytes[i] > SIZE_MAX ||
        (kernel->argument_values[index] & 15U) != 0) return LF_INVALID_ARGUMENT;
  }
  CUdeviceptr output = kernel->argument_values[p[5]];
  for (int i = 0; i < 2; ++i) {
    CUdeviceptr input = kernel->argument_values[p[3+i]];
    if (output <= input ? input-output < bytes[2] : output-input < bytes[i]) {
      return LF_INVALID_ARGUMENT;
    }
  }
  memcpy(kernel->projection, p, sizeof(kernel->projection));
  return LF_OK;
}

static int32_t lf_ordered_projection_prepare(lf_ordered_executor *e) {
  int found = 0;
  uint64_t maximum_bytes = 0;
  for (int i = 0; i < e->kernel_count; ++i) {
    const int32_t *p = e->kernels[i].projection;
    if (p[0] == 0) continue;
    found = 1;
    uint64_t x = (uint64_t)p[0]*p[2]*2;
    uint64_t w = (uint64_t)p[1]*p[2]*2;
    uint64_t y = (uint64_t)p[0]*p[1]*2;
    if (x > 256U*1024U*1024U || w > 256U*1024U*1024U ||
        y > 256U*1024U*1024U) return LF_INVALID_ARGUMENT;
    uint64_t bytes = x+w+y;
    if (bytes > maximum_bytes) maximum_bytes = bytes;
  }
  if (!found) return LF_OK;
  /* A bounded preparation allocation, never request-path scratch growth. */
  if (maximum_bytes > 256U * 1024U * 1024U) return LF_INVALID_ARGUMENT;
#if !defined(__linux__)
  return LF_UNSUPPORTED;
#else
  lf_ordered_projection_state *s = calloc(1, sizeof(*s));
  if (s == NULL) return LF_HOST_ALLOCATION_FAILED;
  e->projections = s;
  s->library = dlopen("libcublas.so.13", RTLD_NOW | RTLD_LOCAL);
  if (s->library == NULL) return LF_UNSUPPORTED;
  int32_t (*create)(void **) = (int32_t (*)(void **))dlsym(s->library, "cublasCreate_v2");
  int32_t (*math)(void *, int) = (int32_t (*)(void *, int))dlsym(s->library, "cublasSetMathMode");
  s->destroy = (int32_t (*)(void *))dlsym(s->library, "cublasDestroy_v2");
  s->set_stream = (int32_t (*)(void *, CUstream))dlsym(s->library, "cublasSetStream_v2");
  s->set_workspace = (int32_t (*)(void *, void *, size_t))dlsym(s->library, "cublasSetWorkspace_v2");
  *(void **)(&s->gemm) = dlsym(s->library, "cublasGemmEx");
  if (!create || !math || !s->destroy || !s->set_stream || !s->set_workspace || !s->gemm) return LF_UNSUPPORTED;
  if (create(&s->handle) != 0 || math(s->handle, 16) != 0) return LF_DRIVER_FAILURE;
  int32_t status = lf_cuda_map_result(e->context->api->cuMemAlloc(
    &s->workspace, LF_ORDERED_PROJECTION_WORKSPACE));
  if (status != LF_OK) return status;
  if (s->set_stream(s->handle, (CUstream)e->stream->handle) != 0 ||
      s->set_workspace(s->handle, (void *)(uintptr_t)s->workspace,
        LF_ORDERED_PROJECTION_WORKSPACE) != 0) return LF_DRIVER_FAILURE;
  status = lf_cuda_map_result(e->context->api->cuMemAlloc(&s->scratch, (size_t)maximum_bytes));
  if (status != LF_OK) return status;
  /* Prime algorithms on private zero-filled operands, never model/KV state. */
  void *zeros = calloc(1, (size_t)maximum_bytes);
  if (zeros == NULL) return LF_HOST_ALLOCATION_FAILED;
  status = lf_cuda_map_result(e->context->api->cuMemcpyHtoD(s->scratch, zeros, (size_t)maximum_bytes));
  free(zeros);
  if (status != LF_OK) return status;
  for (int i = 0; i < e->kernel_count; ++i) {
    const int32_t *p = e->kernels[i].projection;
    if (p[0] == 0) continue;
    int repeated = 0;
    for (int j = 0; j < i; ++j) {
      if (memcmp(p, e->kernels[j].projection, 3*sizeof(int32_t)) == 0) repeated = 1;
    }
    if (repeated) continue;
    CUdeviceptr x = s->scratch, w = x + (uint64_t)p[0]*p[2]*2;
    CUdeviceptr y = w + (uint64_t)p[1]*p[2]*2;
    status = lf_ordered_projection_run(e, p, x, w, y);
    /* Even a failed vendor call can have submitted work. */
    int32_t sync = lf_cuda_map_result(e->context->api->cuStreamSynchronize((CUstream)e->stream->handle));
    if (sync != LF_OK) { atomic_store(&e->phase, LF_ORDERED_ENQUEUED); return sync; }
    if (status != LF_OK) return status;
  }
  status = lf_cuda_map_result(e->context->api->cuMemFree(s->scratch));
  if (status == LF_OK) s->scratch = 0;
  return status;
#endif
}

int32_t lf_ordered_launch_record(lf_ordered_executor *e, lf_ordered_kernel *k) {
  const int32_t *p = k->projection;
  if (p[0] != 0) return lf_ordered_projection_run(e, p,
    k->argument_values[p[3]], k->argument_values[p[4]], k->argument_values[p[5]]);
  return lf_cuda_map_result(e->context->api->cuLaunchKernel(k->function->handle,
    (uint32_t)k->dimensions[0], (uint32_t)k->dimensions[1], (uint32_t)k->dimensions[2],
    (uint32_t)k->dimensions[3], (uint32_t)k->dimensions[4], (uint32_t)k->dimensions[5],
    (uint32_t)k->dimensions[6], (CUstream)e->stream->handle, k->kernel_parameters, NULL));
}
