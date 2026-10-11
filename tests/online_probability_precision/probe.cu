// Independent host recurrence checks the actual generated kernels. This is
// test code, not a replacement model/reference or a production dependency.
#include "main.cuh"
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <cmath>
#define CUDA(call) do { cudaError_t e=(call); if(e!=cudaSuccess){std::fprintf(stderr,"%s\n",cudaGetErrorString(e));std::exit(2);} } while(0)
template<class T> T *buffer(size_t n) { T *p; CUDA(cudaMallocManaged(&p,n*sizeof(T))); return p; }
static float rounded(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }
static float oracle(const __nv_bfloat16 *q,const std::vector<const __nv_bfloat16*> &keys,float sink,int col) {
  std::vector<float> scores(keys.size(),-INFINITY),weights(keys.size());
  for(size_t i=0;i<keys.size();++i) if(keys[i]) {
    float dot=0;
    for(int k=0;k<8;++k){volatile float product=__bfloat162float(q[k])*__bfloat162float(keys[i][k]);dot=dot+product;}
    scores[i]=dot/std::sqrt(8.0f);
  }
  float maximum=-INFINITY,denominator=0,numerator=0;
  for(size_t begin=0;begin<keys.size();begin+=64) {
    const size_t end=std::min(keys.size(),begin+64);
    float next=maximum;
    for(size_t i=begin;i<end;++i)next=std::max(next,scores[i]);
    if(next==-INFINITY)continue;
    const float alpha=maximum==-INFINITY?0:std::exp(maximum-next);
    float sum=0;
    for(size_t i=begin;i<end;++i){const float p=std::exp(scores[i]-next);sum+=p;weights[i]=rounded(p);}
    numerator*=alpha;
    for(size_t i=begin;i<end;++i)if(keys[i])numerator=std::fma(weights[i],__bfloat162float(keys[i][col]),numerator);
    denominator=denominator*alpha+sum; maximum=next;
  }
  return rounded(numerator/(denominator+std::exp(sink-maximum)));
}
int main() {
  auto *append=buffer<int>(3),*counts=buffer<int>(144),*selected=buffer<int>(288),*offsets=buffer<int>(144);
  auto *q=buffer<__nv_bfloat16>(144*2*8),*current=buffer<__nv_bfloat16>(144*8),*ring=buffer<__nv_bfloat16>(128*8),*compressed=buffer<__nv_bfloat16>(128*8),*out=buffer<__nv_bfloat16>(144*2*8);
  auto *sink=buffer<float>(2);sink[0]=-2;sink[1]=6;
  for(int i=0;i<144*2*8;++i)q[i]=__float2bfloat16_rn(float((i*7)%19-9)*0.13f);
  for(int i=0;i<144*8;++i)current[i]=__float2bfloat16_rn(float((i*11)%37-18)*0.27f);
  for(int i=0;i<128*8;++i){ring[i]=__float2bfloat16_rn(float((i*13)%41-20)*0.19f);compressed[i]=__float2bfloat16_rn(float((i*17)%43-21)*0.23f);}
  int checked=0;
  for(auto geometry:std::vector<std::pair<int,int>>{{0,3},{0,66},{0,144},{126,3},{256,1}}) {
    const int base=geometry.first,live=geometry.second;append[0]=base;append[1]=live;append[2]=0;
    for(int row=0;row<144;++row){counts[row]=std::min(2,(base+row+1)/4);selected[row*2]=0;selected[row*2+1]=std::max(0,(base+row+1)/4-1);offsets[row]=0;}
    for(int mode=0;mode<3;++mode) {
      if(mode==0)online_window_attention<<<dim3(144,2),128>>>(append,q,current,ring,sink,out);
      else if(mode==1)online_learned_attention<<<dim3(144,2),128>>>(append,counts,selected,offsets,q,current,ring,compressed,sink,out);
      else online_all_attention<<<dim3(144,2),128>>>(append,counts,selected,offsets,q,current,ring,compressed,sink,out);
      CUDA(cudaGetLastError());CUDA(cudaDeviceSynchronize());
      for(int row=0;row<144;++row)for(int head=0;head<2;++head)for(int col=0;col<8;++col) {
        const int at=(row*2+head)*8+col;const float actual=__bfloat162float(out[at]);
        if(row>=live){if(actual!=0){std::fputs("inactive row not cleared\n",stderr);return 3;}continue;}
        const int position=base+row,first=std::max(0,position-127),length=position-first+1,span=base==0?std::min(128,live):128;
        std::vector<const __nv_bfloat16*> keys(span,nullptr);
        for(int i=0;i<length;++i){int absolute=first+i;keys[i]=absolute>=base?current+(absolute-base)*8:ring+(absolute%128)*8;}
        if(mode) {
          const int visible=(position+1)/4,total=(base+live)/4,padded=mode==1?std::min(2,total):total;
          for(int slot=0;slot<padded;++slot)keys.push_back(slot<(mode==1?counts[row]:visible)?compressed+(mode==1?selected[row*2+slot]:slot)*8:nullptr);
        }
        const float expected=oracle(q+(row*2+head)*8,keys,sink[head],col);
        if(!std::isfinite(actual)||std::abs(actual-expected)>std::max(0.00002f,0.008f*std::abs(expected))){std::fprintf(stderr,"mismatch base=%d live=%d mode=%d row=%d head=%d col=%d actual=%.9g expected=%.9g\n",base,live,mode,row,head,col,actual,expected);return 4;}
        ++checked;
      }
    }
  }
  // Error/empty frame must clear all owned outputs without reading operands.
  append[2]=1;online_window_attention<<<dim3(144,2),128>>>(append,q,current,ring,sink,out);CUDA(cudaDeviceSynchronize());
  for(int i=0;i<144*2*8;++i)if(__bfloat162float(out[i])!=0)return 5;
  for(void *p:{(void*)append,(void*)counts,(void*)selected,(void*)offsets,(void*)q,(void*)current,(void*)ring,(void*)compressed,(void*)out,(void*)sink})CUDA(cudaFree(p));
  std::printf("online_bf16_probability_passed outputs=%d window_learned_all=true masked_boundaries=true ring_wrap=true sink_after_blocks=true inactive_zero=true tolerance=one_bf16_ulp_not_bitwise_tilelang\n",checked);
}
