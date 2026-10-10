// Independent F32 control / transposed residual oracle, not full-model evidence.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "hyper_f32_kernels.cuh"

static void check(cudaError_t e) {
  if (e != cudaSuccess) { std::fprintf(stderr,"%s\n",cudaGetErrorString(e)); std::exit(2); }
}
template<class T> static T* allocate(size_t n) {
  T* p=nullptr; check(cudaMallocManaged(&p,n*sizeof(T))); std::memset(p,0,n*sizeof(T)); return p;
}
static float bf(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }
static float sig(float x) { return x>=0.f ? 1.f/(1.f+std::exp(-x)) : std::exp(x)/(1.f+std::exp(x)); }
static void close_float(float actual,float expected) {
  if (!std::isfinite(actual) || std::fabs(actual-expected)>4.e-6f*(1.f+std::fabs(expected))) {
    std::fprintf(stderr,"float %.9g != %.9g\n",actual,expected); std::exit(3);
  }
}
static void close_bf(float actual,float expected) {
  if (!std::isfinite(actual) || std::fabs(actual-expected)>0.00390625f*(1.f+std::fabs(expected))) {
    std::fprintf(stderr,"bf16 %.9g != %.9g\n",actual,expected); std::exit(4);
  }
}

int main() {
  constexpr int R=3,H=6,S=4,F=24,M=24;
  auto counts=allocate<int>(5);
  auto x=allocate<__nv_bfloat16>(R*F), collapsed=allocate<__nv_bfloat16>(R*H);
  auto norm=allocate<__nv_bfloat16>(R*H), nw=allocate<__nv_bfloat16>(H);
  auto output=allocate<__nv_bfloat16>(R*F), branch=allocate<__nv_bfloat16>(R*H);
  auto weights=allocate<float>(M*F),base=allocate<float>(M),scale=allocate<float>(3);
  auto post=allocate<float>(R*S),initial=allocate<float>(R*S*S),comb=allocate<float>(R*S*S);
  for(int i=0;i<M*F;++i) weights[i]=float((i*7)%31-15)*0.03713f;
  for(int i=0;i<M;++i) base[i]=float((i*3)%17-8)*0.1167f;
  scale[0]=0.711f;scale[1]=1.073f;scale[2]=1.341f;
  for(int i=0;i<H;++i) nw[i]=__float2bfloat16_rn(0.75f+float(i)*0.125f);
  int cases=0,orientation_witnesses=0;
  for(int replay=0;replay<2;++replay) for(int live:{0,1,3,2,3}) {
    counts[3]=live;
    for(int i=0;i<R*F;++i) x[i]=__float2bfloat16_rn(float((i*5+cases*3)%29-14)*0.125f);
    for(int i=0;i<R*H;++i) branch[i]=__float2bfloat16_rn(float((i*11+cases*7)%23-11)*0.0625f);
    for(int i=0;i<R*F;++i) output[i]=__float2bfloat16_rn(17.f);
    hyper_f32_function<<<R,1>>>(counts,x,weights,base,scale,post,initial,collapsed);
    hyper_f32_sinkhorn<<<R,1>>>(counts,initial,comb);
    hyper_f32_norm<<<R,256>>>(counts,collapsed,nw,norm);
    hyper_f32_post<<<R,256>>>(counts,branch,x,post,comb,output);
    check(cudaGetLastError());check(cudaDeviceSynchronize());
    for(int row=0;row<live;++row) {
      float ss=0.f;for(int c=0;c<F;++c) { float v=__bfloat162float(x[row*F+c]);ss=ss+v*v; }
      float inv=1.f/std::sqrt(ss/float(F)+0.00001f),mix[M],pre[S],p[S],a[S*S];
      for(int j=0;j<M;++j) { float dot=0.f;for(int c=0;c<F;++c) dot=dot+__bfloat162float(x[row*F+c])*weights[j*F+c];mix[j]=dot*inv; }
      for(int s=0;s<S;++s) { pre[s]=sig(mix[s]*scale[0]+base[s])+0.000001f;p[s]=2.f*sig(mix[S+s]*scale[1]+base[S+s]);close_float(post[row*S+s],p[s]); }
      for(int i=0;i<S;++i) {
        float max=mix[2*S+i*S]*scale[2]+base[2*S+i*S];
        for(int j=1;j<S;++j) max=std::fmax(max,mix[2*S+i*S+j]*scale[2]+base[2*S+i*S+j]);
        float sum=0.f;for(int j=0;j<S;++j) { a[i*S+j]=std::exp(mix[2*S+i*S+j]*scale[2]+base[2*S+i*S+j]-max);sum=sum+a[i*S+j]; }
        for(int j=0;j<S;++j) { a[i*S+j]=a[i*S+j]/sum+0.000001f;close_float(initial[row*S*S+i*S+j],a[i*S+j]); }
      }
      for(int iteration=0;iteration<20;++iteration) {
        if(iteration) for(int i=0;i<S;++i) { float sum=0.f;for(int j=0;j<S;++j) sum=sum+a[i*S+j];for(int j=0;j<S;++j) a[i*S+j]=a[i*S+j]/(sum+0.000001f); }
        for(int j=0;j<S;++j) { float sum=0.f;for(int i=0;i<S;++i) sum=sum+a[i*S+j];for(int i=0;i<S;++i) a[i*S+j]=a[i*S+j]/(sum+0.000001f); }
      }
      for(int i=0;i<S*S;++i) close_float(comb[row*S*S+i],a[i]);
      float reduced[H];ss=0.f;
      for(int c=0;c<H;++c) { float sum=0.f;for(int s=0;s<S;++s) sum=sum+pre[s]*__bfloat162float(x[row*F+s*H+c]);reduced[c]=bf(sum);close_bf(__bfloat162float(collapsed[row*H+c]),reduced[c]);ss=ss+reduced[c]*reduced[c]; }
      inv=1.f/std::sqrt(ss/float(H)+0.00001f);
      for(int c=0;c<H;++c) close_bf(__bfloat162float(norm[row*H+c]),bf((reduced[c]*inv)*__bfloat162float(nw[c])));
      for(int destination=0;destination<S;++destination) for(int c=0;c<H;++c) {
        float residual=0.f,wrong=0.f;
        for(int source=0;source<S;++source) { float v=__bfloat162float(x[row*F+source*H+c]);residual=residual+a[source*S+destination]*v;wrong=wrong+a[destination*S+source]*v; }
        float product=p[destination]*__bfloat162float(branch[row*H+c]);
        float expected=bf(product+residual);
        close_bf(__bfloat162float(output[row*F+destination*H+c]),expected);
        if(bf(product+wrong)!=expected) ++orientation_witnesses;
      }
    }
    for(int row=live;row<R;++row) for(int c=0;c<F;++c) if(__bfloat162float(output[row*F+c])!=17.f) std::exit(5);
    ++cases;
  }
  if(!orientation_witnesses) std::exit(6);
  for(void *p:{(void*)counts,(void*)x,(void*)collapsed,(void*)norm,(void*)nw,(void*)output,(void*)branch,(void*)weights,(void*)base,(void*)scale,(void*)post,(void*)initial,(void*)comb}) check(cudaFree(p));
  std::printf("F32 hyper cases=%d projection-order=passed transposed-residual=passed witnesses=%d replay=passed inactive=preserved release=balanced\n",cases,orientation_witnesses);
}
