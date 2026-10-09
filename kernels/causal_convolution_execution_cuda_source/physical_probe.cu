#include "convolution_kernels.cuh"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static void check(cudaError_t e) {
  if(e!=cudaSuccess) { std::fprintf(stderr,"%s\n",cudaGetErrorString(e)); std::exit(1); }
}
template<class T> static T* allocate(size_t n) {
  T *p=nullptr; check(cudaMalloc(&p,n*sizeof(T))); return p;
}
template<class T> static void upload(T* p,const std::vector<T>& x) {
  check(cudaMemcpy(p,x.data(),x.size()*sizeof(T),cudaMemcpyHostToDevice));
}
template<class T> static std::vector<T> download(T* p,size_t n) {
  std::vector<T> x(n); check(cudaMemcpy(x.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost)); return x;
}
using B=__nv_bfloat16;

struct Probe {
  int h,d,c,k;
  const void *conv,*delta;
  int *counts,*offsets,*slots,*reset;
  B *input[3],*weight[3],*history[3],*filtered[3],*beta,*output;
  float *decay,*state;
  std::vector<B> x[3],w[3],past[3],b;
  std::vector<float> g;
  std::vector<double> memory;
  Probe(int heads,int dimension,int kernel,const void* cf,const void* df):
    h(heads),d(dimension),c(heads*dimension),k(kernel),conv(cf),delta(df) {
    counts=allocate<int>(5); offsets=allocate<int>(3); slots=allocate<int>(2); reset=allocate<int>(2);
    for(int t=0;t<3;++t) {
      input[t]=allocate<B>(8*c); weight[t]=allocate<B>(c*k);
      history[t]=allocate<B>(4*c*(k-1)); filtered[t]=allocate<B>(8*c);
      x[t].resize(8*c); w[t].resize(c*k); past[t].resize(4*c*(k-1));
      for(size_t i=0;i<x[t].size();++i) x[t][i]=__float2bfloat16((int((i+t)%13)-6)*0.0625f);
      for(size_t i=0;i<w[t].size();++i) w[t][i]=__float2bfloat16((int((i+t)%7)-3)*0.125f);
      for(size_t i=0;i<past[t].size();++i) past[t][i]=__float2bfloat16((int(i%5)-2)*0.25f);
      upload(input[t],x[t]); upload(weight[t],w[t]); upload(history[t],past[t]);
    }
    beta=allocate<B>(8*h); output=allocate<B>(8*c); decay=allocate<float>(8*c); state=allocate<float>(4*c*d);
    b.resize(8*h); g.resize(8*c); memory.resize(4*c*d);
    for(size_t i=0;i<b.size();++i) b[i]=__float2bfloat16((i%4)*0.25f);
    for(size_t i=0;i<g.size();++i) g[i]=-0.01f*(1+i%17);
    for(size_t i=0;i<memory.size();++i) memory[i]=(int(i%9)-4)*0.00390625;
    upload(beta,b); upload(decay,g); upload(state,std::vector<float>(memory.begin(),memory.end()));
  }
  ~Probe() {
    check(cudaFree(state)); check(cudaFree(decay)); check(cudaFree(output)); check(cudaFree(beta));
    for(int t=0;t<3;++t) { check(cudaFree(filtered[t])); check(cudaFree(history[t])); check(cudaFree(weight[t])); check(cudaFree(input[t])); }
    check(cudaFree(reset)); check(cudaFree(slots)); check(cudaFree(offsets)); check(cudaFree(counts));
  }
  void metadata(int active,int rows,const std::vector<int>& off,const std::vector<int>& sl,const std::vector<int>& rst) {
    upload(counts,std::vector<int>{0,active,active,rows,0}); upload(offsets,off); upload(slots,sl); upload(reset,rst);
  }
  void launch() {
    for(int t=0;t<3;++t) {
      void *args[]={&counts,&offsets,&slots,&reset,&input[t],&weight[t],&history[t],&filtered[t]};
      check(cudaLaunchKernel(conv,dim3((c+127)/128,2),dim3(128),args,0,nullptr));
    }
    void *args[]={&counts,&offsets,&slots,&reset,&filtered[0],&filtered[1],&filtered[2],&decay,&beta,&state,&output};
    check(cudaLaunchKernel(delta,dim3(2,h),dim3((d+31)/32*32),args,0,nullptr));
  }
  double correctness() {
    double maxerr=0;
    for(int step=0;step<4;++step) {
      const int active=step==3?0:(step==2?1:2), rows=step==3?0:(step==2?7:8);
      const std::vector<int> off={0,active==2?(step==0?3:1):rows,rows};
      const std::vector<int> sl={step==0?3:1,step==0?1:3},rst={step==0||step==2,step==0};
      metadata(active,rows,off,sl,rst);
      std::vector<B> f[3];
      for(int t=0;t<3;++t) {
        f[t].assign(8*c,__float2bfloat16(-17)); upload(filtered[t],f[t]);
        for(int s=0;s<active;++s) for(int ch=0;ch<c;++ch) {
          size_t base=(size_t(sl[s])*c+ch)*(k-1);
          if(rst[s]) std::fill(past[t].begin()+base,past[t].begin()+base+k-1,__float2bfloat16(0));
          for(int row=off[s];row<off[s+1];++row) {
            double sum=0;
            for(int lag=0;lag<k-1;++lag) sum+=double(__bfloat162float(past[t][base+lag]))*__bfloat162float(w[t][ch*k+lag]);
            B current=x[t][row*c+ch]; sum+=double(__bfloat162float(current))*__bfloat162float(w[t][ch*k+k-1]);
            f[t][row*c+ch]=__float2bfloat16(float(sum/(1+std::exp(-sum))));
            for(int lag=0;lag<k-2;++lag) past[t][base+lag]=past[t][base+lag+1]; past[t][base+k-2]=current;
          }
        }
      }
      std::vector<B> expected(8*c,__float2bfloat16(-17)); upload(output,expected);
      for(int s=0;s<active;++s) {
        size_t base=size_t(sl[s])*c*d;
        if(rst[s]) std::fill(memory.begin()+base,memory.begin()+base+c*d,0);
        for(int row=off[s];row<off[s+1];++row) for(int head=0;head<h;++head) {
          size_t qb=(size_t(row)*h+head)*d,sb=base+size_t(head)*d*d;
          double qs=0,ks=0;
          for(int i=0;i<d;++i) { double q=__bfloat162float(f[0][qb+i]),key=__bfloat162float(f[1][qb+i]); qs+=q*q; ks+=key*key; }
          double qi=1/std::sqrt(qs+0.000001),ki=1/std::sqrt(ks+0.000001);
          for(int col=0;col<d;++col) {
            double read=0;
            for(int i=0;i<d;++i) { memory[sb+size_t(i)*d+col]*=std::exp(double(g[qb+i])); read+=memory[sb+size_t(i)*d+col]*__bfloat162float(f[1][qb+i])*ki; }
            double change=(__bfloat162float(f[2][qb+col])-read)*__bfloat162float(b[row*h+head]),result=0;
            for(int i=0;i<d;++i) { memory[sb+size_t(i)*d+col]+=__bfloat162float(f[1][qb+i])*ki*change; result+=memory[sb+size_t(i)*d+col]*__bfloat162float(f[0][qb+i])*qi/std::sqrt(double(d)); }
            expected[qb+col]=__float2bfloat16(float(result));
          }
        }
      }
      launch(); check(cudaDeviceSynchronize());
      for(int t=0;t<3;++t) {
        auto actual=download(filtered[t],8*c),history_actual=download(history[t],past[t].size());
        if(std::memcmp(history_actual.data(),past[t].data(),past[t].size()*sizeof(B))) { std::fprintf(stderr,"history mismatch step=%d stream=%d\n",step,t); std::exit(2); }
        for(size_t i=0;i<actual.size();++i) maxerr=std::max(maxerr,std::fabs(double(__bfloat162float(actual[i]))-__bfloat162float(f[t][i])));
      }
      auto actual=download(output,8*c); auto after=download(state,memory.size());
      for(size_t i=0;i<actual.size();++i) maxerr=std::max(maxerr,std::fabs(double(__bfloat162float(actual[i]))-__bfloat162float(expected[i])));
      for(size_t i=0;i<after.size();++i) maxerr=std::max(maxerr,std::fabs(double(after[i])-memory[i]));
      if(maxerr>0.003) { std::fprintf(stderr,"oracle error %.9g\n",maxerr); std::exit(2); }
    }
    return maxerr;
  }
  void chunks() {
    // One eight-token prefill must equal eight single-token decode frames,
    // including both raw convolution histories and persistent F32 delta state.
    metadata(1,8,{0,8,8},{2,0},{1,0}); launch(); check(cudaDeviceSynchronize());
    auto expected=download(output,8*c); auto expected_state=download(state,4*c*d);
    std::vector<B> expected_history[3]; for(int t=0;t<3;++t) expected_history[t]=download(history[t],4*c*(k-1));
    std::vector<B> actual;
    for(int row=0;row<8;++row) {
      for(int t=0;t<3;++t) upload(input[t],std::vector<B>(x[t].begin()+row*c,x[t].begin()+(row+1)*c));
      upload(beta,std::vector<B>(b.begin()+row*h,b.begin()+(row+1)*h));
      upload(decay,std::vector<float>(g.begin()+row*c,g.begin()+(row+1)*c));
      metadata(1,1,{0,1,1},{2,0},{row==0,0}); launch(); check(cudaDeviceSynchronize());
      auto part=download(output,c); actual.insert(actual.end(),part.begin(),part.end());
    }
    auto actual_state=download(state,4*c*d);
    if(std::memcmp(actual.data(),expected.data(),8*c*sizeof(B))||std::memcmp(actual_state.data(),expected_state.data(),4*c*d*sizeof(float))) { std::fprintf(stderr,"chunk state/output mismatch\n"); std::exit(2); }
    for(int t=0;t<3;++t) {
      auto actual_history=download(history[t],4*c*(k-1));
      if(std::memcmp(actual_history.data(),expected_history[t].data(),actual_history.size()*sizeof(B))) { std::fprintf(stderr,"chunk history mismatch\n"); std::exit(2); }
    }
  }
};

int main(int argc,char**argv) {
  const int shape=argc>1?std::atoi(argv[1]):0;
  int h,d,k; const void *conv,*delta;
  if(shape==0) { h=2;d=4;k=2;conv=(const void*)conv_small;delta=(const void*)delta_conv_small; }
  else if(shape==1) { h=1;d=129;k=4;conv=(const void*)conv_tail;delta=(const void*)delta_conv_tail; }
  else if(shape==2) { h=64;d=128;k=4;conv=(const void*)conv_glm;delta=(const void*)delta_conv_glm; }
  else return 2;
  Probe probe(h,d,k,conv,delta); double error=probe.correctness(); probe.chunks();
  std::printf("correctness=passed shape=%d channels=%d kernel=%d oracle_maxabs=%.9g history_bitwise=true prefill_decode_chunks_bitwise=true slot-reorder/reset/continuation/idle\n",shape,h*d,k,error);
}
