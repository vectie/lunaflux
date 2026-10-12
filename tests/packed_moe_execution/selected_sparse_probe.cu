// Offline correctness/timing harness. Runtime/IR do not depend on this file.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
struct Case {
  int q,h,kv,d,v,w,t,live,grid,block;
  size_t scratch;
  const void *old_kernel, *new_kernel;
};
#include "selected_sparse.cuh"
static void ck(cudaError_t e) {
  if(e != cudaSuccess) { fprintf(stderr,"CUDA %s\n",cudaGetErrorString(e)); exit(2); }
}
template<class T> struct Device {
  T* p=nullptr;
  size_t count;
  explicit Device(size_t n):count(n) { ck(cudaMalloc(&p,n*sizeof(T))); }
  void put(const std::vector<T>& x) { ck(cudaMemcpy(p,x.data(),count*sizeof(T),cudaMemcpyHostToDevice)); }
  std::vector<T> get() { std::vector<T> x(count); ck(cudaMemcpy(x.data(),p,count*sizeof(T),cudaMemcpyDeviceToHost)); return x; }
  ~Device() { ck(cudaFree(p)); }
};
static std::vector<__nv_bfloat16> varied(size_t n, int seed) {
  std::vector<__nv_bfloat16> x(n);
  for(size_t i=0;i<n;++i) x[i]=__float2bfloat16_rn(float(int((i*17+seed)%63)-31)/32.0f);
  return x;
}
int main(int argc,char**argv) {
  const bool sanitizer=argc>1 && !strcmp(argv[1],"sanitizer");
  const bool analytic=argc>1 && !strcmp(argv[1],"analytic");
  for(size_t c=0;c<sizeof(cases)/sizeof(cases[0]);++c) {
    const Case a=cases[c];
    Device<int> counts(5),positions(a.q),key_count(1),valid(a.t),indices(size_t(a.q)*a.w);
    Device<__nv_bfloat16> q(size_t(a.q)*a.h*a.d),k(size_t(a.t)*a.kv*a.d),v(size_t(a.t)*a.kv*a.v);
    Device<__nv_bfloat16> old(size_t(a.q)*a.h*a.v),out(size_t(a.q)*a.h*a.v);
    Device<unsigned char> scratch(a.scratch);
    q.put(varied(q.count,3)); k.put(varied(k.count,11)); v.put(varied(v.count,23));
    void *old_args[]={&counts.p,&positions.p,&key_count.p,&valid.p,&q.p,&k.p,&v.p,&indices.p,&old.p};
    void *new_args[]={&counts.p,&positions.p,&key_count.p,&valid.p,&q.p,&k.p,&v.p,&indices.p,&out.p,&scratch.p};
    auto launch=[&](bool fresh) {
      ck(cudaLaunchKernel(fresh?a.new_kernel:a.old_kernel,dim3(fresh?a.grid:1),dim3(fresh?a.block:1),fresh?new_args:old_args,0,nullptr));
    };
    auto setup=[&](int mode) {
      std::vector<int> x(size_t(a.q)*a.w,-1),p(a.q,a.live-1),ok(a.t,1);
      std::vector<int> ct={0,0,1,a.q,0};
      for(int row=0;row<a.q;++row) for(int item=0;item<std::min(a.live,a.w);++item)
        x[size_t(row)*a.w+size_t(item)*a.w/std::min(a.live,a.w)]=a.live-1-item;
      if(mode==1) std::fill(x.begin(),x.end(),-1);
      if(mode==2) x[1]=x[0]; // duplicate, entire first query must be zero
      if(mode==3) x[0]=-2;
      if(mode==4) x[0]=a.t;
      if(mode==5) p[0]=0; // selected future keys
      if(mode==6) ok[a.live-1]=0;
      if(mode==7) ct[2]=2;
      if(mode==8) ct[3]=0;
      if(mode==9) ct[3]=a.q-1;
      if(mode==10) {
        std::fill(x.begin(),x.end(),-1);
        for(int row=0;row<a.q;++row) x[size_t(row)*a.w]=a.live-1;
      }
      counts.put(ct); positions.put(p); key_count.put({a.live}); valid.put(ok); indices.put(x);
      ck(cudaMemset(out.p,0xa5,out.count*sizeof(__nv_bfloat16)));
      ck(cudaMemset(scratch.p,0xa5,scratch.count));
    };
    if(analytic) {
      setup(10); launch(true); ck(cudaDeviceSynchronize());
      auto result=out.get(); auto values=v.get();
      for(int row=0;row<a.q;++row) for(int head=0;head<a.h;++head) for(int col=0;col<a.v;++col) {
        const size_t dst=(size_t(row)*a.h+head)*a.v+col;
        const size_t src=(size_t(a.live-1)*a.kv+head/(a.h/a.kv))*a.v+col;
        // A single finite key has probability exactly one, independent of QK.
        if(std::memcmp(&result[dst],&values[src],sizeof(__nv_bfloat16))) return 6;
      }
      printf("analytic case=%zu single_key_copies_value=passed cells=%zu\n",c,result.size());
      continue;
    }
    for(int mode=0;mode<10;++mode) {
      // Full-size valid serial cases are expensive; malformed rows are cheap.
      setup(mode); launch(true); ck(cudaDeviceSynchronize());
      auto result=out.get();
      for(auto value:result) if(!std::isfinite(__bfloat162float(value))) return 3;
      if(!sanitizer) {
        launch(false); ck(cudaDeviceSynchronize()); auto reference=old.get();
        size_t differences=0;
        for(size_t i=0;i<result.size();++i)
          if(std::memcmp(&result[i],&reference[i],sizeof(__nv_bfloat16))) ++differences;
        printf("correctness case=%zu mode=%d exact=%zu/%zu\n",c,mode,result.size()-differences,result.size());
        if(differences) return 4;
      }
      if(mode>=1 && mode<=8) {
        size_t stop=(mode==1 || mode==7 || mode==8)?result.size():size_t(a.h)*a.v;
        for(size_t i=0;i<stop;++i) if(__bfloat162float(result[i])!=0.0f) return 5;
      }
    }
    if(!sanitizer) {
      setup(0); launch(true); launch(false); ck(cudaDeviceSynchronize());
      cudaEvent_t start,end; ck(cudaEventCreate(&start)); ck(cudaEventCreate(&end));
      for(int trial=0;trial<5;++trial) {
        float time[2]={};
        for(int order=0;order<2;++order) {
          bool fresh=((trial+order)%2)==1;
          int repeats=fresh?20:1;
          ck(cudaEventRecord(start)); for(int r=0;r<repeats;++r) launch(fresh);
          ck(cudaEventRecord(end)); ck(cudaEventSynchronize(end));
          ck(cudaEventElapsedTime(&time[fresh?1:0],start,end)); time[fresh?1:0]/=repeats;
        }
        printf("timing case=%zu live=%d q=%d heads=%d d=%d v=%d slots=%d grid=%d block=%d scratch=%zu trial=%d serial_ms=%.6f retained_ms=%.6f\n",
          c,a.live,a.q,a.h,a.d,a.v,a.w,a.grid,a.block,a.scratch,trial,time[0],time[1]); fflush(stdout);
      }
      ck(cudaEventDestroy(end)); ck(cudaEventDestroy(start));
    }
  }
  ck(cudaDeviceReset()); printf("closed=true result=passed sanitizer_only=%d\n",sanitizer); return 0;
}
