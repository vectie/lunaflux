// Offline CUDA capability query. Limits enter the compiler as data; no model,
// scheduler or request-path code includes these CUDA-specific properties.
#include <cuda_runtime.h>
#include <cstdio>
int main() {
  cudaDeviceProp p{};
  if (cudaGetDeviceProperties(&p, 0) != cudaSuccess) return 1;
  std::printf("shared_static_bytes=%zu\nshared_optin_bytes=%zu\nshared_per_unit_bytes=%zu\nregisters_per_unit=%d\nthreads_per_unit=%d\nunits=%d\n",
    p.sharedMemPerBlock, p.sharedMemPerBlockOptin, p.sharedMemPerMultiprocessor,
    p.regsPerMultiprocessor, p.maxThreadsPerMultiProcessor, p.multiProcessorCount);
  return cudaDeviceReset() == cudaSuccess ? 0 : 2;
}
