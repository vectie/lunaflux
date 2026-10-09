// Diagnostic calibration only. No production runtime imports this file.
// CUDA events measure prewarmed graph replays, excluding compilation/allocation.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

static void check(cudaError_t code) {
  if (code != cudaSuccess) {
    std::fprintf(stderr, "CUDA: %s\n", cudaGetErrorString(code));
    std::exit(1);
  }
}
static void blas(cublasStatus_t code) {
  if (code != CUBLAS_STATUS_SUCCESS) {
    std::fprintf(stderr, "cuBLAS: %d\n", int(code));
    std::exit(1);
  }
}
__global__ void fill_float(float *p, size_t n) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += size_t(blockDim.x) * gridDim.x) p[i] = 0.125f;
}
__global__ void fill_bf16(__nv_bfloat16 *p, size_t n) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += size_t(blockDim.x) * gridDim.x) p[i] = __float2bfloat16(1.f / 64.f);
}
__global__ void read_vectors(const float4 *p, float *out, size_t n) {
  const size_t t = blockIdx.x * blockDim.x + threadIdx.x;
  float sum = 0;
  for (size_t i = t; i < n; i += size_t(blockDim.x) * gridDim.x) {
    const float4 v = p[i];
    sum += v.x + v.y + v.z + v.w;
  }
  out[t] = sum;
}
template<class F> static std::vector<double> measure(cudaStream_t stream, F fn,
                                                    int repeats) {
  for (int i = 0; i < 5; ++i) fn();
  check(cudaStreamSynchronize(stream));
  cudaGraph_t graph;
  cudaGraphExec_t exec;
  check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
  for (int i = 0; i < repeats; ++i) fn();
  check(cudaStreamEndCapture(stream, &graph));
  check(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
  cudaEvent_t start, end;
  check(cudaEventCreate(&start));
  check(cudaEventCreate(&end));
  check(cudaGraphLaunch(exec, stream));
  check(cudaStreamSynchronize(stream));
  std::vector<double> us;
  for (int trial = 0; trial < 7; ++trial) {
    check(cudaEventRecord(start, stream));
    check(cudaGraphLaunch(exec, stream));
    check(cudaEventRecord(end, stream));
    check(cudaEventSynchronize(end));
    float ms;
    check(cudaEventElapsedTime(&ms, start, end));
    us.push_back(double(ms) * 1000 / repeats);
  }
  check(cudaEventDestroy(start));
  check(cudaEventDestroy(end));
  check(cudaGraphExecDestroy(exec));
  check(cudaGraphDestroy(graph));
  return us;
}
static double print_samples(const std::vector<double>& us) {
  auto sorted = us;
  std::sort(sorted.begin(), sorted.end());
  std::printf("\"samples_us\":[");
  for (size_t i = 0; i < us.size(); ++i)
    std::printf("%s%.6f", i ? "," : "", us[i]);
  std::printf("],\"median_us\":%.6f", sorted[sorted.size()/2]);
  return sorted[sorted.size()/2];
}
static void memory_case(cudaStream_t stream, size_t bytes, int blocks) {
  float *a, *b, *sums;
  check(cudaMalloc(&a, bytes));
  check(cudaMalloc(&b, bytes));
  check(cudaMalloc(&sums, size_t(blocks) * 256 * sizeof(float)));
  fill_float<<<blocks,256,0,stream>>>(a, bytes / sizeof(float));
  check(cudaGetLastError());
  auto read = measure(stream, [&] {
    read_vectors<<<blocks,256,0,stream>>>(reinterpret_cast<float4*>(a), sums,
                                        bytes / sizeof(float4));
  }, 100);
  std::vector<float> host(size_t(blocks)*256);
  check(cudaMemcpy(host.data(), sums, host.size()*sizeof(float), cudaMemcpyDeviceToHost));
  for (size_t i=0; i<host.size(); ++i) {
    const size_t n = bytes/sizeof(float4);
    const size_t count = i<n ? (n-1-i)/host.size()+1 : 0;
    if (host[i] != float(count)*0.5f) std::exit(2);
  }
  std::printf("{\"test\":\"read\",\"bytes\":%zu,\"blocks\":%d,",bytes,blocks);
  double us = print_samples(read);
  std::printf(",\"GB_s\":%.6f,\"checked\":true}\n",double(bytes)/us/1000);
  auto copy = measure(stream,[&] {
    check(cudaMemcpyAsync(b,a,bytes,cudaMemcpyDeviceToDevice,stream));
  },100);
  float first, last;
  check(cudaMemcpy(&first,b,sizeof(float),cudaMemcpyDeviceToHost));
  check(cudaMemcpy(&last,b+bytes/sizeof(float)-1,sizeof(float),cudaMemcpyDeviceToHost));
  if (first != 0.125f || last != 0.125f) std::exit(2);
  std::printf("{\"test\":\"copy\",\"bytes\":%zu,\"traffic_bytes\":%zu,",bytes,2*bytes);
  us=print_samples(copy);
  std::printf(",\"GB_s_read_plus_write\":%.6f,\"endpoint_checked\":true}\n",2.*bytes/us/1000);
  check(cudaFree(sums)); check(cudaFree(b)); check(cudaFree(a));
}
static void gemm_case(cublasHandle_t handle, cudaStream_t stream,
                      int m, int n, int k) {
  __nv_bfloat16 *x, *w, *y;
  check(cudaMalloc(&x,size_t(m)*k*2));
  check(cudaMalloc(&w,size_t(n)*k*2));
  check(cudaMalloc(&y,size_t(m)*n*2));
  fill_bf16<<<288,256,0,stream>>>(x,size_t(m)*k);
  fill_bf16<<<288,256,0,stream>>>(w,size_t(n)*k);
  check(cudaGetLastError());
  const float alpha=1, beta=0;
  auto times=measure(stream,[&] {
    blas(cublasGemmEx(handle,CUBLAS_OP_T,CUBLAS_OP_N,n,m,k,&alpha,
         w,CUDA_R_16BF,k,x,CUDA_R_16BF,k,&beta,y,CUDA_R_16BF,n,
         CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  },30);
  std::vector<__nv_bfloat16> host(size_t(m)*n);
  check(cudaMemcpy(host.data(),y,host.size()*2,cudaMemcpyDeviceToHost));
  const float expected=__bfloat162float(__float2bfloat16(float(k)/4096.f));
  for (auto v:host) if (__bfloat162float(v)!=expected) std::exit(2);
  const double flops=2.*m*n*k;
  const size_t bytes=2*(size_t(m)*k+size_t(n)*k+size_t(m)*n);
  std::printf("{\"test\":\"gemm\",\"m\":%d,\"n\":%d,\"k\":%d,\"flops\":%.0f,\"minimum_bytes\":%zu,",m,n,k,flops,bytes);
  const double us=print_samples(times);
  std::printf(",\"TFLOP_s\":%.6f,\"all_outputs_checked\":true}\n",flops/us/1e6);
  check(cudaFree(y)); check(cudaFree(w)); check(cudaFree(x));
}
int main() {
  check(cudaSetDevice(0));
  cudaDeviceProp p;
  check(cudaGetDeviceProperties(&p,0));
  char pci[32]; check(cudaDeviceGetPCIBusId(pci,sizeof(pci),0));
  std::printf("{\"test\":\"device\",\"name\":\"%s\",\"pci\":\"%s\",\"sm_count\":%d,\"l2_bytes\":%d}\n",p.name,pci,p.multiProcessorCount,p.l2CacheSize);
  cudaStream_t stream; check(cudaStreamCreate(&stream));
  for (int blocks : {p.multiProcessorCount*4,p.multiProcessorCount*8}) {
    memory_case(stream,4ULL<<20,blocks);
    memory_case(stream,256ULL<<20,blocks);
  }
  cublasHandle_t handle; blas(cublasCreate(&handle));
  blas(cublasSetStream(handle,stream));
  int version; blas(cublasGetVersion(handle,&version));
  std::printf("{\"test\":\"library\",\"cublas_version\":%d}\n",version);
  gemm_case(handle,stream,4096,4096,4096);
  for (int m : {1,8,16,128,1528,2048}) {
    gemm_case(handle,stream,m,4096,1024);
    gemm_case(handle,stream,m,1024,2048);
    gemm_case(handle,stream,m,6144,1024);
    gemm_case(handle,stream,m,1024,3072);
  }
  blas(cublasDestroy(handle)); check(cudaStreamDestroy(stream));
  std::printf("{\"outcome\":\"complete\",\"scope\":\"calibration-not-serving\"}\n");
}
