#include "recurrent_kernels.cuh"
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

struct Probe {
  int h,k,v,rows,sequences,block;
  const void *fresh,*legacy;
  int *counts,*offsets,*slots,*reset;
  __nv_bfloat16 *q,*key,*value,*beta,*output,*old_output;
  float *decay,*cache,*initial,*final;
  Probe(int heads,int keys,int values,int r,int s,const void* f,const void* old):
    h(heads),k(keys),v(values),rows(r),sequences(s),block((std::max(keys,values)+31)/32*32),fresh(f),legacy(old) {
    counts=allocate<int>(5); offsets=allocate<int>(s+1); slots=allocate<int>(s); reset=allocate<int>(s);
    q=allocate<__nv_bfloat16>(r*h*k); key=allocate<__nv_bfloat16>(r*h*k);
    value=allocate<__nv_bfloat16>(r*h*v); beta=allocate<__nv_bfloat16>(r*h);
    output=allocate<__nv_bfloat16>(r*h*v); old_output=allocate<__nv_bfloat16>(r*h*v);
    decay=allocate<float>(r*h*k); cache=allocate<float>(4*h*k*v);
    initial=allocate<float>(s*h*k*v); final=allocate<float>(s*h*k*v);
  }
  ~Probe() {
    check(cudaFree(final)); check(cudaFree(initial)); check(cudaFree(cache)); check(cudaFree(decay));
    check(cudaFree(old_output)); check(cudaFree(output)); check(cudaFree(beta)); check(cudaFree(value));
    check(cudaFree(key)); check(cudaFree(q)); check(cudaFree(reset)); check(cudaFree(slots));
    check(cudaFree(offsets)); check(cudaFree(counts));
  }
  void launch(bool old) {
    void *new_args[]={&counts,&offsets,&slots,&reset,&q,&key,&value,&decay,&beta,&cache,&output};
    void *old_args[]={&counts,&offsets,&q,&key,&value,&decay,&beta,&initial,&old_output,&final};
    check(cudaLaunchKernel(old?legacy:fresh,old?dim3(1):dim3(sequences,h),old?dim3(1):dim3(block),old?old_args:new_args,0,nullptr));
  }
  float timed(bool old,int repeats) {
    cudaEvent_t a,b; check(cudaEventCreate(&a)); check(cudaEventCreate(&b));
    check(cudaEventRecord(a)); for(int i=0;i<repeats;++i) launch(old);
    check(cudaEventRecord(b)); check(cudaEventSynchronize(b)); float ms;
    check(cudaEventElapsedTime(&ms,a,b)); check(cudaEventDestroy(b)); check(cudaEventDestroy(a));
    return ms*1000/repeats;
  }
  double correctness(bool compare_serial=true) {
    std::vector<__nv_bfloat16> x(rows*h*k),y(rows*h*k),z(rows*h*v),b(rows*h);
    std::vector<float> g(rows*h*k),state(4*h*k*v);
    for(size_t i=0;i<x.size();++i) { x[i]=__float2bfloat16((int(i%11)-5)*0.0625f); y[i]=__float2bfloat16((int(i%7)-3)*0.09375f); g[i]=-0.01f*(1+i%17); }
    for(size_t i=0;i<z.size();++i) z[i]=__float2bfloat16((int(i%13)-6)*0.03125f);
    for(size_t i=0;i<b.size();++i) b[i]=__float2bfloat16((i%4)*0.25f);
    for(size_t i=0;i<state.size();++i) state[i]=(int(i%9)-4)*0.00390625f;
    // Exercise zero Q/K norms as well as nonzero vectors.
    for(int i=0;i<k;++i) { x[i]=__float2bfloat16(0); y[i]=__float2bfloat16(0); }
    upload(q,x); upload(key,y); upload(value,z); upload(beta,b); upload(decay,g); upload(cache,state);
    std::vector<double> oracle(state.begin(),state.end());
    double maxerr=0;
    for(int step=0;step<4;++step) {
      int live=step==3?0:(step==2?rows-1:rows), active=step==3?0:(step==2?1:2);
      std::vector<int> off(sequences+1,live),slot(sequences,0),rst(sequences,0);
      off[0]=0; if(active==2) off[1]=step==0?rows/2:1;
      slot[0]=step==0?3:1; if(active==2) slot[1]=step==0?1:3;
      rst[0]=step==0||step==2; if(active==2) rst[1]=step==0;
      upload(counts,std::vector<int>{0,active,active,live,0}); upload(offsets,off); upload(slots,slot); upload(reset,rst);
      auto before=download(cache,state.size());
      std::vector<float> init(sequences*h*k*v,0.0f);
      for(int s=0;s<active;++s) for(int i=0;i<h*k*v;++i) init[s*h*k*v+i]=rst[s]?0.0f:before[slot[s]*h*k*v+i];
      upload(initial,init); upload(final,std::vector<float>(init.size(),-17.0f));
      std::vector<__nv_bfloat16> expected(rows*h*v,__float2bfloat16(-17));
      upload(output,expected); upload(old_output,expected);
      for(int s=0;s<active;++s) {
        size_t base=size_t(slot[s])*h*k*v;
        if(rst[s]) std::fill(oracle.begin()+base,oracle.begin()+base+h*k*v,0.0);
        for(int row=off[s];row<off[s+1];++row) for(int head=0;head<h;++head) {
          size_t qb=(size_t(row)*h+head)*k,vb=(size_t(row)*h+head)*v,sb=base+size_t(head)*k*v;
          double qs=0,ks=0;
          for(int i=0;i<k;++i) { double a=__bfloat162float(x[qb+i]),c=__bfloat162float(y[qb+i]); qs+=a*a; ks+=c*c; }
          double qi=1/std::sqrt(qs+0.000001),ki=1/std::sqrt(ks+0.000001);
          for(int column=0;column<v;++column) {
            double read=0;
            for(int i=0;i<k;++i) { oracle[sb+size_t(i)*v+column]*=std::exp(double(g[qb+i])); read+=oracle[sb+size_t(i)*v+column]*__bfloat162float(y[qb+i])*ki; }
            double delta=(__bfloat162float(z[vb+column])-read)*__bfloat162float(b[row*h+head]),result=0;
            for(int i=0;i<k;++i) { oracle[sb+size_t(i)*v+column]+=__bfloat162float(y[qb+i])*ki*delta; result+=oracle[sb+size_t(i)*v+column]*__bfloat162float(x[qb+i])*qi/std::sqrt(double(k)); }
            expected[vb+column]=__float2bfloat16(float(result));
          }
        }
      }
      if(compare_serial) launch(true);
      launch(false); check(cudaDeviceSynchronize());
      auto actual=download(output,expected.size()),old=download(old_output,expected.size());
      if(compare_serial && std::memcmp(actual.data(),old.data(),actual.size()*sizeof(__nv_bfloat16))) { std::fprintf(stderr,"output bits differ step=%d\n",step); std::exit(2); }
      auto after=download(cache,state.size()),old_state=download(final,init.size());
      for(int s=0;s<active;++s) for(int i=0;i<h*k*v;++i) {
        if(compare_serial && std::memcmp(&after[slot[s]*h*k*v+i],&old_state[s*h*k*v+i],sizeof(float))) { std::fprintf(stderr,"state bits differ step=%d\n",step); std::exit(2); }
      }
      for(size_t i=0;i<actual.size();++i) maxerr=std::max(maxerr,std::fabs(double(__bfloat162float(actual[i]))-__bfloat162float(expected[i])));
      for(size_t i=0;i<after.size();++i) maxerr=std::max(maxerr,std::fabs(double(after[i])-oracle[i]));
      if(maxerr>0.003) { std::fprintf(stderr,"oracle error %.9g\n",maxerr); std::exit(2); }
    }
    // Timing compares fresh requests; both routes start from identical zero state.
    std::vector<int> off(sequences+1,rows),slot(sequences,0),rst(sequences,1); off[0]=0; off[1]=rows/2; slot[1]=1;
    upload(counts,std::vector<int>{0,2,2,rows,0}); upload(offsets,off); upload(slots,slot); upload(reset,rst);
    upload(initial,std::vector<float>(sequences*h*k*v,0.0f));
    return maxerr;
  }
};

int main(int argc,char**argv) {
  const int shape=argc>1?std::atoi(argv[1]):0;
  const bool timing=argc>2 && std::strcmp(argv[2],"timing")==0;
  const bool compare_serial=!(argc>2 && std::strcmp(argv[2],"sanitize")==0);
  int h,k,v,r,s; const void *fresh,*old;
  if(shape==0) { h=2;k=4;v=6;r=4;s=2;fresh=(const void*)delta_small;old=(const void*)delta_small_legacy; }
  else if(shape==1) { h=4;k=32;v=16;r=8;s=4;fresh=(const void*)delta_medium;old=(const void*)delta_medium_legacy; }
  else if(shape==2) { h=64;k=128;v=128;r=8;s=2;fresh=(const void*)delta_glm;old=(const void*)delta_glm_legacy; }
  else return 2;
  Probe probe(h,k,v,r,s,fresh,old); double error=probe.correctness(compare_serial);
  std::printf("correctness=passed shape=%d heads=%d keys=%d values=%d slot-reorder/reset/continuation/idle bitwise_output_state=%s oracle_maxabs=%.9g\n",shape,h,k,v,compare_serial?"true":"not-tested",error);
  if(timing) {
    probe.launch(true);probe.launch(false);check(cudaDeviceSynchronize());
    for(int trial=0;trial<5;++trial) {
      float a,b;
      const int repeats=shape==2?3:10;
      if(trial%2==0) { a=probe.timed(true,repeats);b=probe.timed(false,repeats); }
      else { b=probe.timed(false,repeats);a=probe.timed(true,repeats); }
      std::printf("tokens=%d rows=2 history=0 trial=%d old_us=%.9g new_us=%.9g bitwise=true maxabs=0 oracle_maxabs=%.9g\n",r,trial,a,b,error);
    }
  }
}
