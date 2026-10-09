#include "block_kernels.cuh"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
using B=__nv_bfloat16;
static void check(cudaError_t e) { if(e!=cudaSuccess) { std::fprintf(stderr,"%s\n",cudaGetErrorString(e)); std::exit(1); } }
template<class T> static T* alloc(size_t n) { T*p=nullptr;check(cudaMalloc(&p,n*sizeof(T)));return p; }
template<class T> static void put(T*p,const std::vector<T>&v) { check(cudaMemcpy(p,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice)); }
template<class T> static std::vector<T> get(T*p,size_t n) { std::vector<T>v(n);check(cudaMemcpy(v.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost));return v; }
static B bf_round(double v) { return __float2bfloat16_rn(float(v)); }
static double f(B v) { return __bfloat162float(v); }
static std::vector<B> dense(const std::vector<B>&x,const std::vector<B>&w,int rows,int in,int out) {
  std::vector<B>y(rows*out); for(int r=0;r<rows;++r)for(int o=0;o<out;++o) {
    double sum=0;for(int k=0;k<in;++k)sum+=f(x[r*in+k])*f(w[o*in+k]); y[r*out+o]=bf_round(sum);
  }return y;
}
struct Probe {
  static constexpr int H=2,D=4,C=8,F=6,R=3,K=4,N=8,S=4;
  int *counts,*offsets,*slots,*reset;
  B *hidden,*output,*weights[12],*history[3],*filtered[3],*frame[12];
  float *bias,*alog,*decay,*state;
  std::vector<B>x,w[12],past[3]; std::vector<double>memory;
  Probe() {
    counts=alloc<int>(5);offsets=alloc<int>(3);slots=alloc<int>(2);reset=alloc<int>(2);
    hidden=alloc<B>(N*F);output=alloc<B>(N*F);bias=alloc<float>(C);alog=alloc<float>(H);
    decay=alloc<float>(N*C);state=alloc<float>(S*C*D);memory.resize(S*C*D);
    x.resize(N*F);for(size_t i=0;i<x.size();++i)x[i]=bf_round((int(i%11)-5)*0.125);put(hidden,x);
    int sizes[]={C*F,C*F,C*F,R*F,C*R,H*F,R*F,C*R,D,F*C,C*K,C*K};
    for(int j=0;j<12;++j) { w[j].resize(sizes[j]);weights[j]=alloc<B>(sizes[j]);
      for(int i=0;i<sizes[j];++i)w[j][i]=bf_round(j==8?1.0:(int((i+j)%7)-3)*0.125);put(weights[j],w[j]); }
    for(int j=0;j<3;++j) {history[j]=alloc<B>(S*C*(K-1));filtered[j]=alloc<B>(N*C);past[j].resize(S*C*(K-1));put(history[j],past[j]);}
    int sizes_f[]={N*C,N*C,N*C,N*R,N*C,N*H,N*H,1,N*R,N*C,N*C,N*C};
    for(int j=0;j<12;++j)frame[j]=alloc<B>(sizes_f[j]);
    put(bias,std::vector<float>(C,0.125f));put(alog,std::vector<float>(H,-1.0f));put(state,std::vector<float>(memory.size()));
  }
  ~Probe(){ for(auto p:frame)check(cudaFree(p));for(auto p:weights)check(cudaFree(p));
    for(int i=0;i<3;++i){check(cudaFree(history[i]));check(cudaFree(filtered[i]));}
    check(cudaFree(state));check(cudaFree(decay));check(cudaFree(alog));check(cudaFree(bias));check(cudaFree(output));check(cudaFree(hidden));
    check(cudaFree(reset));check(cudaFree(slots));check(cudaFree(offsets));check(cudaFree(counts)); }
  void projection(const void*kernel,B*input,B*weight,B*out,int width) {
    void*args[]={&counts,&input,&weight,&out};check(cudaLaunchKernel(kernel,dim3((N*width+255)/256),dim3(256),args,0,nullptr));
  }
  void launch() {
    projection((void*)block_0,hidden,weights[0],frame[0],C);projection((void*)block_1,hidden,weights[1],frame[1],C);
    projection((void*)block_2,hidden,weights[2],frame[2],C);projection((void*)block_3,hidden,weights[3],frame[3],R);
    projection((void*)block_4,frame[3],weights[4],frame[4],C);projection((void*)block_5,hidden,weights[5],frame[5],H);
    void*ba[]={&counts,&frame[5],&frame[6]};check(cudaLaunchKernel((void*)block_6,dim3(1),dim3(256),ba,0,nullptr));
    void*da[]={&counts,&frame[4],&bias,&alog,&decay};check(cudaLaunchKernel((void*)block_7,dim3(1),dim3(256),da,0,nullptr));
    projection((void*)block_8,hidden,weights[6],frame[8],R);projection((void*)block_9,frame[8],weights[7],frame[9],C);
    for(int j=0;j<3;++j){ B*weight=weights[j==0?10:11];void*args[]={&counts,&offsets,&slots,&reset,&frame[j],&weight,&history[j],&filtered[j]};check(cudaLaunchKernel((void*)block_conv,dim3(1,2),dim3(128),args,0,nullptr)); }
    void*ra[]={&counts,&offsets,&slots,&reset,&filtered[0],&filtered[1],&filtered[2],&decay,&frame[6],&state,&frame[10]};
    check(cudaLaunchKernel((void*)block_delta,dim3(2,H),dim3(32),ra,0,nullptr));
    void*na[]={&counts,&frame[10],&frame[9],&weights[8],&frame[11]};check(cudaLaunchKernel((void*)block_14,dim3(N,H),dim3(128),na,0,nullptr));
    projection((void*)block_15,frame[11],weights[9],output,F);
  }
  std::vector<B> oracle(int rows,int active,const std::vector<int>&off,const std::vector<int>&sl,const std::vector<int>&rst) {
    std::vector<B>qkv[3];for(int j=0;j<3;++j)qkv[j]=dense(x,w[j],rows,F,C);
    auto raw=dense(dense(x,w[3],rows,F,R),w[4],rows,R,C),b=dense(x,w[5],rows,F,H);
    auto gate=dense(dense(x,w[6],rows,F,R),w[7],rows,R,C);
    std::vector<double>g(rows*C);for(int i=0;i<rows*C;++i)g[i]=-5.0/(1+std::exp(-std::exp(-1.0)*(f(raw[i])+0.125)));
    for(auto&v:b)v=bf_round(1/(1+std::exp(-f(v))));
    for(int j=0;j<3;++j)for(int s=0;s<active;++s)for(int ch=0;ch<C;++ch) {
      size_t base=(sl[s]*C+ch)*(K-1);if(rst[s])std::fill(past[j].begin()+base,past[j].begin()+base+K-1,bf_round(0));
      for(int r=off[s];r<off[s+1];++r) {auto&cw=w[j==0?10:11];double sum=0;for(int k=0;k<K-1;++k)sum+=f(past[j][base+k])*f(cw[ch*K+k]);
        B current=qkv[j][r*C+ch];sum+=f(current)*f(cw[ch*K+K-1]);qkv[j][r*C+ch]=bf_round(sum/(1+std::exp(-sum)));
        for(int k=0;k<K-2;++k)past[j][base+k]=past[j][base+k+1];past[j][base+K-2]=current; }
    }
    std::vector<B>y(rows*C);
    for(int s=0;s<active;++s) {int sb=sl[s]*C*D;if(rst[s])std::fill(memory.begin()+sb,memory.begin()+sb+C*D,0);
      for(int r=off[s];r<off[s+1];++r)for(int h=0;h<H;++h) {int qb=(r*H+h)*D,base=sb+h*D*D;double qs=0,ks=0;
        for(int k=0;k<D;++k){qs+=f(qkv[0][qb+k])*f(qkv[0][qb+k]);ks+=f(qkv[1][qb+k])*f(qkv[1][qb+k]);}
        double qi=1/std::sqrt(qs+1e-6),ki=1/std::sqrt(ks+1e-6);
        for(int v=0;v<D;++v){double read=0;for(int k=0;k<D;++k){memory[base+k*D+v]*=std::exp(g[qb+k]);read+=memory[base+k*D+v]*f(qkv[1][qb+k])*ki;}
          double change=(f(qkv[2][qb+v])-read)*f(b[r*H+h]),result=0;
          for(int k=0;k<D;++k){memory[base+k*D+v]+=f(qkv[1][qb+k])*ki*change;result+=memory[base+k*D+v]*f(qkv[0][qb+k])*qi/std::sqrt(double(D));}y[qb+v]=bf_round(result); }
      }
    }
    for(int r=0;r<rows;++r)for(int h=0;h<H;++h){int base=(r*H+h)*D;double sum=0;for(int k=0;k<D;++k)sum+=f(y[base+k])*f(y[base+k]);
      double inverse=1/std::sqrt(sum/D+1e-6);for(int k=0;k<D;++k)y[base+k]=bf_round(f(y[base+k])*inverse*f(w[8][k])/(1+std::exp(-f(gate[base+k]))));}
    return dense(y,w[9],rows,C,F);
  }
  void meta(int active,int rows,const std::vector<int>&off,const std::vector<int>&sl,const std::vector<int>&rst) {
    put(counts,std::vector<int>{0,active,active,rows,0});put(offsets,off);put(slots,sl);put(reset,rst);
  }
  void test() {
    double maxerr=0;
    for(int step=0;step<3;++step){int rows=step==2?7:8,active=step==2?1:2;
      std::vector<int>off={0,active==2?(step==0?3:1):rows,rows},sl={step==0?3:1,step==0?1:3},rst={step==0||step==2,step==0};
      meta(active,rows,off,sl,rst);auto expected=oracle(rows,active,off,sl,rst);launch();check(cudaDeviceSynchronize());auto actual=get(output,rows*F);
      for(size_t i=0;i<actual.size();++i)maxerr=std::max(maxerr,std::fabs(f(actual[i])-f(expected[i])));
      for(int j=0;j<3;++j){auto got=get(history[j],S*C*(K-1));if(std::memcmp(got.data(),past[j].data(),got.size()*sizeof(B))) {std::fprintf(stderr,"history mismatch\n");std::exit(2);}}
      auto got=get(state,memory.size());for(size_t i=0;i<got.size();++i)maxerr=std::max(maxerr,std::fabs(double(got[i])-memory[i]));
    }
    if(maxerr>0.008){std::fprintf(stderr,"oracle error %.9g\n",maxerr);std::exit(2);}
    // Full prefill versus eight decode frames must have identical stage rounds.
    put(hidden,x);meta(1,8,{0,8,8},{2,0},{1,0});launch();check(cudaDeviceSynchronize());
    auto expected=get(output,N*F);auto expected_state=get(state,S*C*D);std::vector<B>actual;
    for(int r=0;r<N;++r){put(hidden,std::vector<B>(x.begin()+r*F,x.begin()+(r+1)*F));meta(1,1,{0,1,1},{2,0},{r==0,0});launch();check(cudaDeviceSynchronize());auto part=get(output,F);actual.insert(actual.end(),part.begin(),part.end());}
    auto actual_state=get(state,S*C*D);
    if(std::memcmp(actual.data(),expected.data(),actual.size()*sizeof(B))||std::memcmp(actual_state.data(),expected_state.data(),actual_state.size()*sizeof(float))){std::fprintf(stderr,"chunk mismatch\n");std::exit(2);}
    std::printf("complete_recurrent_branch=passed launches=16 oracle_maxabs=%.9g history_bitwise=true prefill_decode_bitwise=true\n",maxerr);
  }
};
int main(){Probe probe;probe.test();}
