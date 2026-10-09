#include "indexed_sparse.cuh"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>

static void ck(cudaError_t error) {
  if (error != cudaSuccess) { std::fprintf(stderr, "%s\n", cudaGetErrorString(error)); std::exit(2); }
}
template<class T> static T *buf(size_t count) {
  T *p; ck(cudaMalloc(&p, count * sizeof(T))); ck(cudaMemset(p, 0, count * sizeof(T))); return p;
}
template<class T> static void put(T *p, const std::vector<T> &v) {
  ck(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
}
static __nv_bfloat16 bf(float x) { return __float2bfloat16_rn(x); }

int main() {
  int devices = 0; ck(cudaGetDeviceCount(&devices)); if (!devices) return 3;
  ck(cudaSetDevice(0));
  int32_t *hc = buf<int32_t>(5), *qc = buf<int32_t>(5), *dc = buf<int32_t>(5);
  int32_t *pos = buf<int32_t>(8), *dp = buf<int32_t>(1), *valid = buf<int32_t>(32);
  auto *iq = buf<__nv_bfloat16>(32), *iw = buf<__nv_bfloat16>(8);
  auto *ik = buf<__nv_bfloat16>(128), *gate = buf<__nv_bfloat16>(128), *ape = buf<__nv_bfloat16>(16);
  auto *q = buf<__nv_bfloat16>(64), *k = buf<__nv_bfloat16>(128), *v = buf<__nv_bfloat16>(128);
  auto *pooled = buf<__nv_bfloat16>(32);
  int32_t *raw = buf<int32_t>(32), *pv = buf<int32_t>(8), *pc = buf<int32_t>(1);
  // Decode storage is exactly one row, not maximum_history rows. Sanitizers
  // therefore detect either clearing or writing beyond the new query domain.
  int32_t *selected = buf<int32_t>(88), *ds = buf<int32_t>(11);
  auto *out = buf<__nv_bfloat16>(64), *dout = buf<__nv_bfloat16>(8);
  put(hc, std::vector<int32_t>{0,0,1,15,0});
  put(qc, std::vector<int32_t>{0,0,1,3,0});
  put(dc, std::vector<int32_t>{0,0,1,1,0});
  put(pos, std::vector<int32_t>{7,8,14,0,0,0,0,0}); put(dp, std::vector<int32_t>{14});
  std::vector<int32_t> hv(32,0); for (int i=0;i<15;++i) hv[i]=1; put(valid,hv);
  std::vector<__nv_bfloat16> values(128,bf(0));
  for(int token=0;token<15;++token) for(int c=0;c<4;++c) values[token*4+c]=bf((token+1)*(c+1)*0.125f);
  put(v,values);
  sparse_prefill_pool<<<1,1>>>(hc,ik,gate,valid,ape,pooled,raw,pv,pc);
  sparse_prefill_index<<<1,1>>>(qc,pos,iq,iw,pooled,raw,pv,pc,valid,selected);
  sparse_prefill_attention<<<1,1>>>(qc,pos,hc+3,valid,q,k,v,selected,out);
  ck(cudaGetLastError()); ck(cudaDeviceSynchronize());
  std::vector<int32_t> indices(88); std::vector<__nv_bfloat16> output(64);
  ck(cudaMemcpy(indices.data(),selected,352,cudaMemcpyDeviceToHost));
  ck(cudaMemcpy(output.data(),out,128,cudaMemcpyDeviceToHost));
  float max_error=0.0f;
  for(int row=0;row<3;++row) {
    int position = row==0?7:row==1?8:14;
    std::vector<int32_t> expected(11,-1);
    for(int rank=0;rank<8;++rank) expected[rank]=rank;
    int tail=(position+1)%4, start=position+1-tail;
    for(int t=0;t<tail;++t) expected[8+t]=start+t;
    for(int rank=0;rank<11;++rank) if(indices[row*11+rank]!=expected[rank]) return 4;
    float prob=__bfloat162float(bf(1.0f/(8+tail)));
    for(int head=0;head<2;++head) for(int c=0;c<4;++c) {
      float sum=0;
      for(int rank=0;rank<11;++rank) if(expected[rank]>=0) {
        volatile float product=prob*__bfloat162float(values[expected[rank]*4+c]); sum=sum+product;
      }
      float error=std::fabs(__bfloat162float(output[(row*2+head)*4+c])-__bfloat162float(bf(sum)));
      max_error=std::fmax(max_error,error);
    }
  }
  for(int cell=24;cell<64;++cell) if(__bfloat162float(output[cell])!=0.0f) return 5;
  sparse_decode_pool<<<1,1>>>(hc,ik,gate,valid,ape,pooled,raw,pv,pc);
  sparse_decode_index<<<1,1>>>(dc,dp,iq,iw,pooled,raw,pv,pc,valid,ds);
  sparse_decode_attention<<<1,1>>>(dc,dp,hc+3,valid,q,k,v,ds,dout);
  ck(cudaGetLastError()); ck(cudaDeviceSynchronize());
  std::vector<__nv_bfloat16> decoded(8); std::vector<int32_t> dids(11);
  ck(cudaMemcpy(decoded.data(),dout,16,cudaMemcpyDeviceToHost));
  ck(cudaMemcpy(dids.data(),ds,44,cudaMemcpyDeviceToHost));
  if(std::memcmp(decoded.data(),output.data()+16,16)!=0 ||
     std::memcmp(dids.data(),indices.data()+22,44)!=0) return 6;
  put(dc,std::vector<int32_t>{0,0,1,0,0});
  sparse_decode_index<<<1,1>>>(dc,dp,iq,iw,pooled,raw,pv,pc,valid,ds);
  sparse_decode_attention<<<1,1>>>(dc,dp,hc+3,valid,q,k,v,ds,dout);
  ck(cudaGetLastError()); ck(cudaDeviceSynchronize());
  ck(cudaMemcpy(decoded.data(),dout,16,cudaMemcpyDeviceToHost));
  for(auto x:decoded) if(__bfloat162float(x)!=0) return 7;
  put(dc,std::vector<int32_t>{0,0,1,1,0});
  cudaEvent_t begin,end; ck(cudaEventCreate(&begin)); ck(cudaEventCreate(&end));
  ck(cudaEventRecord(begin));
  for(int repeat=0;repeat<64;++repeat) {
    sparse_decode_pool<<<1,1>>>(hc,ik,gate,valid,ape,pooled,raw,pv,pc);
    sparse_decode_index<<<1,1>>>(dc,dp,iq,iw,pooled,raw,pv,pc,valid,ds);
    sparse_decode_attention<<<1,1>>>(dc,dp,hc+3,valid,q,k,v,ds,dout);
  }
  ck(cudaEventRecord(end)); ck(cudaEventSynchronize(end));
  float elapsed=0; ck(cudaEventElapsedTime(&elapsed,begin,end));
  ck(cudaEventDestroy(begin)); ck(cudaEventDestroy(end));
  for(void *p : std::vector<void*>{hc,qc,dc,pos,dp,valid,iq,iw,ik,gate,ape,q,k,v,pooled,raw,pv,pc,selected,ds,out,dout}) ck(cudaFree(p));
  ck(cudaDeviceSynchronize());
  std::printf("maxabs=%.9g prefill_decode_bitwise=true idle_output_zero=true all_allocations_released=true fixture_decode_3stage_us=%.3f\n",max_error,elapsed*1000.0f/64);
  return max_error==0.0f?0:8;
}
