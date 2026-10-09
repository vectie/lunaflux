// Small independent CPU oracle for the generated learned text suffix.
// Not a real checkpoint, full decoder or performance qualification.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "learned_text_io_kernels.cuh"

static void check(cudaError_t e) {
  if (e != cudaSuccess) { std::fprintf(stderr, "%s\n", cudaGetErrorString(e)); std::exit(2); }
}
template<class T> static T* allocate(size_t n) {
  T* p = nullptr;
  check(cudaMallocManaged(&p, n * sizeof(T)));
  std::memset(p, 0, n * sizeof(T));
  return p;
}
static float bf(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }
static float sigmoid(float x) {
  if (x >= 0) return 1.f / (1.f + std::exp(-x));
  float e = std::exp(x); return e / (1.f + e);
}
int main() {
  constexpr int H=8, S=4, V=17, R=5, O=2, F=S*H;
  auto counts=allocate<int32_t>(5), outputs=allocate<int32_t>(1);
  auto selected=allocate<int32_t>(O), tokens=allocate<int32_t>(O);
  auto residual=allocate<__nv_bfloat16>(R*F);
  auto function=allocate<float>(S*F), base=allocate<float>(S), scale=allocate<float>(1);
  auto norm=allocate<__nv_bfloat16>(H), head=allocate<__nv_bfloat16>(V*H);
  auto collapse=allocate<__nv_bfloat16>(R*H), hidden=allocate<__nv_bfloat16>(O*H);
  auto logits=allocate<float>(O*V);
  for (int i=0;i<R*F;++i) residual[i]=__float2bfloat16_rn(float((i*5)%29-14)*0.125f);
  for (int i=0;i<S*F;++i) function[i]=float((i*3)%11-5)*0.015625f;
  for (int s=0;s<S;++s) base[s]=float(s-1)*0.25f;
  scale[0]=0.75f;
  for (int c=0;c<H;++c) norm[c]=__float2bfloat16_rn(0.5f+c*0.0625f);
  for (int i=0;i<V*H;++i) head[i]=__float2bfloat16_rn(float((i*7)%23-11)*0.0625f);
  auto run=[&]() {
    learned_fixture_stream_reduce<<<R,256>>>(counts,residual,function,base,scale,collapse);
    learned_fixture_norm<<<O,256>>>(counts,outputs,selected,collapse,norm,hidden);
    learned_fixture_head<<<1,256>>>(outputs,hidden,head,logits);
    learned_fixture_greedy<<<O,1>>>(counts,outputs,selected,logits,tokens);
    check(cudaGetLastError()); check(cudaDeviceSynchronize());
  };
  for (int live : {1,3,5}) {
    counts[3]=live; outputs[0]=live==1 ? 1 : 2;
    selected[0]=live-1; selected[1]=0;
    float reference[R*H]={}, expected_hidden[O*H]={}, expected_logits[O*V]={};
    int expected_tokens[O]={};
    for (int row=0;row<live;++row) {
      float squares=0;
      for (int c=0;c<F;++c) { float x=__bfloat162float(residual[row*F+c]); squares=squares+x*x; }
      float inverse=1.f/std::sqrt(squares/float(F)+1.e-5f), controls[S];
      for (int s=0;s<S;++s) {
        float dot=0;
        for (int c=0;c<F;++c) dot=dot+__bfloat162float(residual[row*F+c])*function[s*F+c];
        controls[s]=sigmoid((dot*inverse)*scale[0]+base[s])+1.e-6f;
      }
      for (int c=0;c<H;++c) {
        float sum=0;
        for (int s=0;s<S;++s) sum=sum+controls[s]*__bfloat162float(residual[row*F+s*H+c]);
        reference[row*H+c]=bf(sum);
      }
    }
    for (int row=0;row<outputs[0];++row) {
      float squares=0;
      for (int c=0;c<H;++c) { float x=reference[selected[row]*H+c]; squares=squares+x*x; }
      float inverse=1.f/std::sqrt(squares/float(H)+1.e-5f);
      for (int c=0;c<H;++c) expected_hidden[row*H+c]=bf(bf(reference[selected[row]*H+c]*inverse)*__bfloat162float(norm[c]));
      float best=-INFINITY; expected_tokens[row]=-1;
      for (int w=0;w<V;++w) {
        float sum=0;
        for (int c=0;c<H;++c) sum=sum+expected_hidden[row*H+c]*__bfloat162float(head[w*H+c]);
        expected_logits[row*V+w]=bf(sum);
        if (expected_tokens[row]<0 || expected_logits[row*V+w]>best) { best=expected_logits[row*V+w]; expected_tokens[row]=w; }
      }
    }
    for (int i=0;i<R*H;++i) collapse[i]=__float2bfloat16_rn(19.f);
    for (int i=0;i<O*H;++i) hidden[i]=__float2bfloat16_rn(21.f);
    for (int i=0;i<O*V;++i) logits[i]=23.f;
    tokens[0]=tokens[1]=123;
    __nv_bfloat16 first_collapse[R*H], first_hidden[O*H]; float first_logits[O*V];
    for (int repeat=0;repeat<2;++repeat) {
      run();
      float error=0;
      for (int i=0;i<live*H;++i) error=std::max(error,std::fabs(__bfloat162float(collapse[i])-reference[i]));
      for (int i=0;i<outputs[0]*H;++i) error=std::max(error,std::fabs(__bfloat162float(hidden[i])-expected_hidden[i]));
      for (int i=0;i<outputs[0]*V;++i) error=std::max(error,std::fabs(logits[i]-expected_logits[i]));
      if (error>0.03125f) return 3;
      for (int i=0;i<outputs[0];++i) if (tokens[i]!=expected_tokens[i]) return 4;
      for (int i=live*H;i<R*H;++i) if (__bfloat162float(collapse[i])!=19.f) return 5;
      for (int i=outputs[0]*H;i<O*H;++i) if (__bfloat162float(hidden[i])!=21.f) return 6;
      for (int i=outputs[0]*V;i<O*V;++i) if (logits[i]!=23.f) return 7;
      for (int i=outputs[0];i<O;++i) if (tokens[i]!=123) return 8;
      if (repeat==0) {
        std::memcpy(first_collapse,collapse,sizeof(first_collapse));
        std::memcpy(first_hidden,hidden,sizeof(first_hidden));
        std::memcpy(first_logits,logits,sizeof(first_logits));
      } else if (std::memcmp(first_collapse,collapse,sizeof(first_collapse)) ||
                 std::memcmp(first_hidden,hidden,sizeof(first_hidden)) ||
                 std::memcmp(first_logits,logits,sizeof(first_logits))) return 9;
      std::printf("live=%d repeat=%d maxabs=%.9g greedy=exact inactive=untouched\n",live,repeat,error);
    }
  }
  counts[3]=R; outputs[0]=O; selected[0]=4; selected[1]=1;
  // Invalid controls must overwrite previous successful data, not sample it.
  const float original_base=base[0], original_scale=scale[0], original_function=function[0];
  const auto original_residual=residual[0];
  for (int invalid=0;invalid<4;++invalid) {
    base[0]=original_base; scale[0]=original_scale; function[0]=original_function; residual[0]=original_residual;
    if (invalid==0) base[0]=NAN;
    if (invalid==1) scale[0]=NAN;
    if (invalid==2) function[0]=NAN;
    if (invalid==3) residual[0]=__float2bfloat16_rn(NAN);
    selected[0]=0; selected[1]=0;
    run();
    for (int c=0;c<H;++c) if (!std::isnan(__bfloat162float(collapse[c]))) return 10;
    if (tokens[0]!=-1 || tokens[1]!=-1) return 11;
  }
  outputs[0]=0; tokens[0]=123; tokens[1]=124; run();
  if (tokens[0]!=123 || tokens[1]!=124) return 12;
  check(cudaFree(logits)); check(cudaFree(hidden)); check(cudaFree(collapse));
  check(cudaFree(head)); check(cudaFree(norm)); check(cudaFree(scale));
  check(cudaFree(base)); check(cudaFree(function)); check(cudaFree(residual));
  check(cudaFree(tokens)); check(cudaFree(selected)); check(cudaFree(outputs)); check(cudaFree(counts));
  std::puts("learned_text_io=passed numeric=true deterministic=true invalid_controls=fail_closed all_allocations=released");
  return 0;
}
