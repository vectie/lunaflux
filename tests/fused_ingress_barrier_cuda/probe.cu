#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include KERNEL_SOURCE
#define CK(call) do { cudaError_t e = (call); if (e != cudaSuccess) { std::fprintf(stderr, "%s\n", cudaGetErrorString(e)); std::exit(2); } } while(0)
struct Buffer {
  void *p;
  explicit Buffer(size_t bytes) { CK(cudaMalloc(&p, bytes)); CK(cudaMemset(p, 0, bytes)); }
  ~Buffer() { CK(cudaFree(p)); }
  template<class T> T* as() { return static_cast<T*>(p); }
  void put(const void *h, size_t n) { CK(cudaMemcpy(p, h, n, cudaMemcpyHostToDevice)); }
};
int main() {
  CK(cudaSetDevice(0));
  for (int tokens : {1,2,7,8,15,16,17,31,32}) {
    int counts[5] = {1,0,1,tokens,2}, offsets[2] = {0,tokens}, tables[2] = {0,2}, pages[2] = {0,1};
    std::vector<int> positions(tokens); for(int i=0;i<tokens;++i) positions[i]=i;
    Buffer dc(sizeof counts), dpos(tokens*4), doff(8), dseq(4), dtoff(8), dpage(8);
    dc.put(counts,sizeof counts); dpos.put(positions.data(),tokens*4); doff.put(offsets,8); dseq.put(&tokens,4); dtoff.put(tables,8); dpage.put(pages,8);
    std::vector<__nv_bfloat16> x(tokens*LF_INPUT_WIDTH), w(LF_QUERY_HEADS*LF_HEAD_DIM*LF_INPUT_WIDTH, __float2bfloat16_rn(1.0f/LF_INPUT_WIDTH)), norm(LF_HEAD_DIM,__float2bfloat16_rn(1.0f));
    for(int t=0;t<tokens;++t) for(int k=0;k<LF_INPUT_WIDTH;++k) x[t*LF_INPUT_WIDTH+k]=__float2bfloat16_rn(float(t+1)/8);
    Buffer dx(x.size()*2), dw(w.size()*2), dn(norm.size()*2), dout(tokens*LF_OUTPUT_WIDTH*2);
    Buffer dk(LF_TOTAL_PAGE_COUNT*LF_PAGE_STRIDE_BF16*2), dv(LF_TOTAL_PAGE_COUNT*LF_PAGE_STRIDE_BF16*2);
    dx.put(x.data(),x.size()*2); dw.put(w.data(),w.size()*2); dn.put(norm.data(),norm.size()*2);
    lunaflux_fused_qwen_qkv_qknorm_rope_kvwrite_bf16_head128_production_v2<<<dim3((tokens+15)/16,LF_PACKED_HEADS),128>>>(
      dc.as<int>(),dpos.as<int>(),doff.as<int>(),dseq.as<int>(),dtoff.as<int>(),dpage.as<int>(),
      dx.as<__nv_bfloat16>(),dw.as<__nv_bfloat16>(),dw.as<__nv_bfloat16>(),dw.as<__nv_bfloat16>(),dn.as<__nv_bfloat16>(),dn.as<__nv_bfloat16>(),dout.as<__nv_bfloat16>(),dk.as<__nv_bfloat16>(),dv.as<__nv_bfloat16>());
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    std::vector<__nv_bfloat16> out(tokens*LF_OUTPUT_WIDTH), key(LF_TOTAL_PAGE_COUNT*LF_PAGE_STRIDE_BF16), value(key.size());
    CK(cudaMemcpy(out.data(),dout.p,out.size()*2,cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(key.data(),dk.p,key.size()*2,cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(value.data(),dv.p,value.size()*2,cudaMemcpyDeviceToHost));
    for(int t=0;t<tokens;++t) for(int h=0;h<LF_PACKED_HEADS;++h) for(int c=0;c<LF_HEAD_DIM;++c) {
      float v=float(t+1)/8, ref=v;
      if(h<LF_QK_HEADS) {
        float n=__bfloat162float(__float2bfloat16_rn(v/std::sqrt(v*v+LF_NORM_EPSILON)));
        int pair=c%(LF_HEAD_DIM/2); float angle=float(t)*float(std::pow(1000000.0,-2.0*pair/LF_HEAD_DIM));
        ref=__bfloat162float(__float2bfloat16_rn(n*(std::cos(angle)+(c<LF_HEAD_DIM/2?-1:1)*std::sin(angle))));
      }
      float got=__bfloat162float(out[t*LF_OUTPUT_WIDTH+h*LF_HEAD_DIM+c]);
      if(!std::isfinite(got)||std::fabs(got-ref)>0.02f) { std::fprintf(stderr,"mismatch d=%d t=%d h=%d c=%d got=%g ref=%g\n",LF_HEAD_DIM,t,h,c,got,ref); return 3; }
      if(h>=LF_QUERY_HEADS) {
        int kh=h<LF_QK_HEADS?h-LF_QUERY_HEADS:h-LF_QK_HEADS;
        size_t index=size_t(t/LF_TOKENS_PER_PAGE)*LF_PAGE_STRIDE_BF16+(t%LF_TOKENS_PER_PAGE)*LF_KV_WIDTH+kh*LF_HEAD_DIM+c;
        float cached=__bfloat162float((h<LF_QK_HEADS?key:value)[index]);
        if(cached!=got) return 4;
      }
    }
    std::printf("dimension=%d tokens=%d numeric=passed kv=passed\n",LF_HEAD_DIM,tokens);
  }
}
