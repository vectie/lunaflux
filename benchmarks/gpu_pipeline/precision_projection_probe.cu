#include "precision_kernels.cuh"
#include "precision_fixture.h"
#include <cstdio>
#include <cmath>
#include <vector>
#include <stdexcept>

static void check(cudaError_t r) { if(r!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(r)); }
template<class T> class DeviceArray {
  T *p_=nullptr;
public:
  explicit DeviceArray(size_t n) { check(cudaMalloc(&p_,(n?n:1)*sizeof(T))); }
  ~DeviceArray() { if(p_) cudaFree(p_); }
  DeviceArray(const DeviceArray&)=delete;
  DeviceArray& operator=(const DeviceArray&)=delete;
  T *get() { return p_; }
  void close() { if(p_) { check(cudaFree(p_)); p_=nullptr; } }
};
int main() {
  try {
    std::vector<__nv_bfloat16> host_input(32*128);
    for(int i=0;i<32*128;i++) host_input[i]=__float2bfloat16_rn((float)(i%13-6)*0.125f);
    DeviceArray<__nv_bfloat16> input(32*128),output(32*64);
    DeviceArray<int> counts(4);
    DeviceArray<float> global(1);
    check(cudaMemcpy(input.get(),host_input.data(),host_input.size()*2,cudaMemcpyHostToDevice));
    int successes=0;
    for(const auto &c:projection_cases) {
      DeviceArray<unsigned char> payload(c.payload_bytes),scales(c.scale_bytes);
      check(cudaMemcpy(payload.get(),c.payload,c.payload_bytes,cudaMemcpyHostToDevice));
      if(c.scale_bytes) check(cudaMemcpy(scales.get(),c.scales,c.scale_bytes,cudaMemcpyHostToDevice));
      check(cudaMemcpy(global.get(),&c.global_bits,4,cudaMemcpyHostToDevice));
      for(int rows:{1,3,32}) {
        int host_counts[4]={0,0,rows,rows};
        check(cudaMemcpy(counts.get(),host_counts,16,cudaMemcpyHostToDevice));
        check(cudaMemset(output.get(),0xab,32*64*2));
        c.launch(counts.get(),input.get(),payload.get(),scales.get(),global.get(),output.get());
        check(cudaGetLastError()); check(cudaDeviceSynchronize());
        std::vector<__nv_bfloat16> actual(32*64);
        check(cudaMemcpy(actual.data(),output.get(),actual.size()*2,cudaMemcpyDeviceToHost));
        for(int i=0;i<rows*64;i++) {
          const float value=__bfloat162float(actual[i]),ref=c.expected[i];
          if(!std::isfinite(value) || fabsf(value-ref)>0.025f+fabsf(ref)*0.01f) {
            std::fprintf(stderr,"format=%s rows=%d index=%d reference=%.9g actual=%.9g\n",c.name,rows,i,ref,value);
            throw std::runtime_error("compact projection numerical mismatch");
          }
        }
        for(int i=rows*64;i<32*64;i++) if(__bfloat16_as_ushort(actual[i])!=0xababU) throw std::runtime_error("row tail overwritten");
        std::printf("format=%s rows=%d outcome=passed\n",c.name,rows); ++successes;
      }
    }
    std::printf("outcome=passed cases=%d scope=compact-to-BF16-compiler-projection-not-full-serving\n",successes);
    global.close(); counts.close(); output.close(); input.close();
    check(cudaDeviceReset()); return 0;
  } catch(const std::exception &e) { std::fprintf(stderr,"failure=%s\n",e.what()); return 1; }
}
