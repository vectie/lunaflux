#include "precision_kernels.cuh"
#include "precision_fixture.h"
#include <cstdio>
#include <cstring>
#include <vector>
#include <stdexcept>

static void check(cudaError_t result) {
  if (result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
}
template<class T> class DeviceArray {
  T *ptr_=nullptr;
public:
  explicit DeviceArray(size_t n) { check(cudaMalloc(&ptr_, (n ? n : 1)*sizeof(T))); }
  ~DeviceArray() { if(ptr_) cudaFree(ptr_); }
  DeviceArray(const DeviceArray&)=delete;
  DeviceArray& operator=(const DeviceArray&)=delete;
  T *get() { return ptr_; }
};
static void compare(const char *label,const void *expected,const void *actual,size_t n) {
  if (std::memcmp(expected,actual,n)!=0) {
    const auto *e=(const unsigned char*)expected,*a=(const unsigned char*)actual;
    for(size_t i=0;i<n;i++) if(e[i]!=a[i]) {
      std::fprintf(stderr,"%s mismatch offset=%zu expected=%u actual=%u\n",label,i,e[i],a[i]); break;
    }
    throw std::runtime_error("CPU/GPU scalar contract mismatch");
  }
}
int main() {
  try {
    int count=0;
    for(const auto &c:precision_cases) {
      const size_t elements=(size_t)c.rows*c.columns;
      DeviceArray<float> input(elements),global(1);
      DeviceArray<unsigned char> payload(c.payload_bytes),scales(c.scale_bytes);
      DeviceArray<__nv_bfloat16> output(elements);
      DeviceArray<int> error(1);
      const float one=1.0f;
      check(cudaMemcpy(input.get(),c.input,elements*4,cudaMemcpyHostToDevice));
      check(cudaMemcpy(global.get(),&one,4,cudaMemcpyHostToDevice));
      check(cudaMemset(error.get(),0,4));
      check(cudaMemset(payload.get(),0xab,c.payload_bytes));
      c.launch(input.get(),payload.get(),scales.get(),global.get(),output.get(),error.get());
      check(cudaGetLastError()); check(cudaDeviceSynchronize());
      int err=0; unsigned g=0;
      check(cudaMemcpy(&err,error.get(),4,cudaMemcpyDeviceToHost));
      if(err) throw std::runtime_error("finite fixture rejected");
      check(cudaMemcpy(&g,global.get(),4,cudaMemcpyDeviceToHost));
      compare("global",&c.global_bits,&g,4);
      std::vector<unsigned char> p(c.payload_bytes),s(c.scale_bytes),o(elements*2);
      check(cudaMemcpy(p.data(),payload.get(),p.size(),cudaMemcpyDeviceToHost));
      if(!s.empty()) check(cudaMemcpy(s.data(),scales.get(),s.size(),cudaMemcpyDeviceToHost));
      check(cudaMemcpy(o.data(),output.get(),o.size(),cudaMemcpyDeviceToHost));
      compare("payload",c.payload,p.data(),p.size());
      compare("scales",c.scales,s.data(),s.size());
      compare("reconstructed-bf16",c.expected,o.data(),o.size());
      std::printf("case=%s outcome=passed\n",c.name); ++count;
    }
    std::printf("outcome=passed cases=%d scope=precision-conversion-not-full-serving\n",count);
    check(cudaDeviceReset()); return 0;
  } catch(const std::exception &e) { std::fprintf(stderr,"failure=%s\n",e.what()); return 1; }
}
