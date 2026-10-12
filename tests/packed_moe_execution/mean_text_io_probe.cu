// Small head fixture, not a checkpoint-quality or performance comparison.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "mean_text_io_kernels.cuh"

static void check(cudaError_t e) {
  if(e != cudaSuccess) { std::fprintf(stderr,"%s\n",cudaGetErrorString(e)); std::exit(2); }
}
#include "text_head_timing.cuh"
template<class T> static T* allocate(size_t n) {
  T* p=nullptr; check(cudaMallocManaged(&p,n*sizeof(T))); std::memset(p,0,n*sizeof(T)); return p;
}
static float bf(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }
int main(int argc, char** argv) {
  constexpr int H=8,S=4,V=17,R=5,O=2;
  auto counts=allocate<int32_t>(5), output_count=allocate<int32_t>(1);
  auto selected=allocate<int32_t>(O), tokens=allocate<int32_t>(O);
  auto residual=allocate<__nv_bfloat16>(R*S*H), norm=allocate<__nv_bfloat16>(H);
  auto head=allocate<__nv_bfloat16>(V*H), hidden=allocate<__nv_bfloat16>(O*H);
  auto logits=allocate<float>(O*V);
  for(int i=0;i<R*S*H;++i) residual[i]=__float2bfloat16_rn(float((i*7)%37-18)*0.09375f);
  for(int i=0;i<H;++i) norm[i]=__float2bfloat16_rn(0.75f+i*0.0625f);
  for(int i=0;i<V*H;++i) head[i]=__float2bfloat16_rn(float((i*11)%29-14)*0.078125f);
  auto launch=[&]() {
    text_fixture_norm<<<O,256>>>(counts,output_count,selected,residual,norm,hidden);
    text_fixture_head<<<1,256>>>(output_count,hidden,head,logits);
    text_fixture_greedy<<<O,1>>>(counts,output_count,selected,logits,tokens);
    check(cudaGetLastError());
  };
  auto run=[&]() { launch(); check(cudaDeviceSynchronize()); };
  int rounding_witnesses=0;
  for(int live : {1,3,5}) {
    counts[3]=live; output_count[0]=live==1?1:2;
    selected[0]=live-1; selected[1]=0;
    float expected_hidden[O*H]={}, expected_logits[O*V]={};
    int expected_tokens[O]={};
    for(int r=0;r<output_count[0];++r) {
      float mean[H], squares=0;
      for(int c=0;c<H;++c) {
        float sum=0;
        for(int s=0;s<S;++s) sum=sum+__bfloat162float(residual[(selected[r]*S+s)*H+c]);
        mean[c]=bf(sum/float(S)); squares=squares+mean[c]*mean[c];
      }
      float inverse=1.f/std::sqrt(squares/H+1.e-5f);
      for(int c=0;c<H;++c) {
        float scaled=mean[c]*inverse,w=__bfloat162float(norm[c]);
        expected_hidden[r*H+c]=bf(bf(scaled)*w);
        if(expected_hidden[r*H+c]!=bf(scaled*w)) ++rounding_witnesses;
      }
      float best=-INFINITY; expected_tokens[r]=-1;
      for(int w=0;w<V;++w) {
        float sum=0;
        for(int c=0;c<H;++c) sum=sum+expected_hidden[r*H+c]*__bfloat162float(head[w*H+c]);
        expected_logits[r*V+w]=bf(sum);
        if(expected_tokens[r]<0 || bf(sum)>best) { best=bf(sum); expected_tokens[r]=w; }
      }
    }
    for(int i=0;i<O*H;++i) hidden[i]=__float2bfloat16_rn(21.f);
    for(int i=0;i<O*V;++i) logits[i]=23.f;
    tokens[0]=tokens[1]=123;
    __nv_bfloat16 first_hidden[O*H]; float first_logits[O*V];
    for(int repeat=0;repeat<2;++repeat) {
      run();
      float error=0;
      for(int i=0;i<output_count[0]*H;++i) error=std::max(error,std::fabs(__bfloat162float(hidden[i])-expected_hidden[i]));
      for(int i=0;i<output_count[0]*V;++i) error=std::max(error,std::fabs(logits[i]-expected_logits[i]));
      // Literal fixtures are chosen away from CPU/GPU rsqrt rounding midpoints.
      if(error!=0) return 3;
      for(int i=0;i<output_count[0];++i) if(tokens[i]!=expected_tokens[i]) return 4;
      for(int i=output_count[0]*H;i<O*H;++i) if(__bfloat162float(hidden[i])!=21.f) return 5;
      for(int i=output_count[0]*V;i<O*V;++i) if(logits[i]!=23.f) return 6;
      if(output_count[0]==1 && tokens[1]!=123) return 7;
      if(repeat==0) { std::memcpy(first_hidden,hidden,sizeof(first_hidden)); std::memcpy(first_logits,logits,sizeof(first_logits)); }
      else if(std::memcmp(first_hidden,hidden,sizeof(first_hidden)) || std::memcmp(first_logits,logits,sizeof(first_logits))) return 8;
      std::printf("live=%d repeat=%d maxabs=%.9g selected_rows=matched greedy=exact inactive=untouched\n",live,repeat,error);
    }
  }
  if(argc==2 && std::strcmp(argv[1],"--benchmark")==0) time_text_head(launch);
  output_count[0]=O; selected[0]=-1; selected[1]=R; run();
  if(tokens[0]!=-1 || tokens[1]!=-1) return 9;
  output_count[0]=0; tokens[0]=123; tokens[1]=124; run();
  if(tokens[0]!=123 || tokens[1]!=124) return 10;
  if(!rounding_witnesses) return 11;
  check(cudaFree(logits)); check(cudaFree(hidden)); check(cudaFree(head)); check(cudaFree(norm));
  check(cudaFree(residual)); check(cudaFree(tokens)); check(cudaFree(selected)); check(cudaFree(output_count)); check(cudaFree(counts));
  std::printf("mean_text_io=passed intermediate_rounding_witnesses=%d all_allocations=released\n",rounding_witnesses);
}
