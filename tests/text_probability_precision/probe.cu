// Test-only independent host recurrence; no production ATen dependency.
#include "main.cuh"
#include <vector>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#define CUDA(call) do { auto e=(call); if(e!=cudaSuccess){std::fprintf(stderr,"%s\n",cudaGetErrorString(e));std::exit(2);} } while(0)
template<class T> T *buffer(size_t n) { T *p; CUDA(cudaMallocManaged(&p,n*sizeof(T))); return p; }
static float bf(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }
static float oracle(const __nv_bfloat16 *q,const __nv_bfloat16 *k,const __nv_bfloat16 *v,int row,int head,int col,int block) {
  float maximum=-INFINITY,denominator=0,numerator=0;
  const int qb=(row*4+head)*8,kh=head/2;
  for(int first=0;first<=row;first+=block) {
    const int end=std::min(first+block,row+1);
    std::vector<float> scores(end-first);
    float next=maximum;
    for(int key=first;key<end;++key) {
      float dot=0;
      for(int d=0;d<8;++d) { volatile float p=__bfloat162float(q[qb+d])*__bfloat162float(k[(key*2+kh)*8+d]); dot+=p; }
      scores[key-first]=dot/std::sqrt(8.0f);next=std::max(next,scores[key-first]);
    }
    float alpha=std::exp(maximum-next);numerator*=alpha;denominator*=alpha;
    for(int key=first;key<end;++key) {
      float p=std::exp(scores[key-first]-next);denominator+=p;
      numerator=std::fma(bf(p),__bfloat162float(v[(key*2+kh)*8+col]),numerator);
    }
    maximum=next;
  }
  return bf(numerator/denominator);
}
int main() {
  auto *q=buffer<__nv_bfloat16>(140*4*8),*k=buffer<__nv_bfloat16>(140*2*8),*v=buffer<__nv_bfloat16>(140*2*8),*out=buffer<__nv_bfloat16>(140*4*8);
  int checked=0;
  for(int pattern=0;pattern<3;++pattern) {
    for(int i=0;i<140*4*8;++i) q[i]=__float2bfloat16_rn(pattern==0?0.0f:float((i*7)%19-9)*(pattern==1?0.13f:4.0f));
    for(int i=0;i<140*2*8;++i) {
      k[i]=__float2bfloat16_rn(float((i*11)%37-18)*0.27f);
      v[i]=__float2bfloat16_rn(float((i*13)%41-20)*0.19f);
    }
    for(int block:{64,128}) {
      if(block==64)probability64_attention<<<dim3(140,4),32>>>(q,k,v,out);
      else probability128_attention<<<dim3(140,4),32>>>(q,k,v,out);
      CUDA(cudaGetLastError());CUDA(cudaDeviceSynchronize());
      for(int row=0;row<140;++row)for(int h=0;h<4;++h)for(int col=0;col<8;++col) {
        const float expected=oracle(q,k,v,row,h,col,block),actual=__bfloat162float(out[(row*4+h)*8+col]);
        if(!std::isfinite(actual)||std::abs(actual-expected)>std::max(0.00001f,0.008f*std::abs(expected))) {
          std::fprintf(stderr,"mismatch pattern=%d block=%d row=%d h=%d col=%d actual=%g expected=%g\n",pattern,block,row,h,col,actual,expected);return 3;
        }
        ++checked;
      }
    }
  }
  for(void *p:{(void*)q,(void*)k,(void*)v,(void*)out})CUDA(cudaFree(p));
  std::printf("causal_gqa_probability_pass outputs=%d block64_128=true zero_large_scores=true future_mask=true tolerance=one_bf16_ulp_not_matrix_equivalence\n",checked);
}
