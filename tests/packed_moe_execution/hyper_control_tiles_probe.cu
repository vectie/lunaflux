#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include "hyper_tiles.cuh"

#define CK(x) do { auto e=(x); if(e!=cudaSuccess){fprintf(stderr,"%s: %s\n",#x,cudaGetErrorString(e));exit(2);} } while(0)
using Launch = void(*)(bool,int*,__nv_bfloat16*,__nv_bfloat16*,float*,float*,float*,float*,float*,__nv_bfloat16*);
template<class T> T* allocate(size_t n){T*p; CK(cudaMalloc(&p,n*sizeof(T)));return p;}
template<class T> void upload(T*p,const std::vector<T>&a){CK(cudaMemcpy(p,a.data(),a.size()*sizeof(T),cudaMemcpyHostToDevice));}
template<class T> std::vector<T> read(T*p,size_t n){std::vector<T>a(n);CK(cudaMemcpy(a.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost));return a;}
template<class T> void equal(const std::vector<T>&a,const std::vector<T>&b,const char*name){
  if(a.size()!=b.size()||memcmp(a.data(),b.data(),a.size()*sizeof(T))){
    for(size_t i=0;i<a.size();++i)if(memcmp(&a[i],&b[i],sizeof(T))){fprintf(stderr,"%s differs index=%zu\n",name,i);break;}
    exit(3);
  }
}
void run(int id,int s,int h,int r,Launch launch,bool sanitize){
  int controls=s*(s+2), width=s*h;
  auto counts=allocate<int>(5); auto x=allocate<__nv_bfloat16>(r*width);
  auto w=allocate<__nv_bfloat16>(controls*width);auto base=allocate<float>(controls);auto scale=allocate<float>(3);
  auto scratch=allocate<float>(r*(controls+1));auto post=allocate<float>(r*s);auto matrix=allocate<float>(r*s*s);auto y=allocate<__nv_bfloat16>(r*h);
  std::vector<__nv_bfloat16> hx(r*width),hw(controls*width);
  for(size_t i=0;i<hx.size();++i)hx[i]=__float2bfloat16_rn(float(int((i*37+11)%211)-105)*.0078125f);
  for(size_t i=0;i<hw.size();++i)hw[i]=__float2bfloat16_rn(float(int((i*17+3)%101)-50)*.0009765625f);
  std::vector<float> hb(controls);for(int i=0;i<controls;++i)hb[i]=(i-5)*.03125f;
  upload(x,hx);upload(w,hw);upload(base,hb);upload(scale,std::vector<float>{.125f,.25f,.125f});
  auto invoke=[&](bool p){launch(p,counts,x,w,base,scale,scratch,post,matrix,y);CK(cudaGetLastError());};
  // Live prefix, tails, no work and malformed count retain inactive outputs.
  for(int live: {r,1,r-1,0,r+1}){
    upload(counts,std::vector<int>{1,0,1,live,0});
    CK(cudaMemset(post,0x5a,r*s*4));CK(cudaMemset(matrix,0x5a,r*s*s*4));CK(cudaMemset(y,0x5a,r*h*2));
    invoke(false);CK(cudaDeviceSynchronize());auto op=read(post,r*s);auto om=read(matrix,r*s*s);auto oy=read(y,r*h);
    CK(cudaMemset(post,0x5a,r*s*4));CK(cudaMemset(matrix,0x5a,r*s*s*4));CK(cudaMemset(y,0x5a,r*h*2));
    invoke(true);CK(cudaDeviceSynchronize());equal(op,read(post,r*s),"post");equal(om,read(matrix,r*s*s),"matrix");equal(oy,read(y,r*h),"collapse");
  }
  upload(counts,std::vector<int>{1,0,1,r,0});
  if(!sanitize){
    cudaEvent_t a,b;CK(cudaEventCreate(&a));CK(cudaEventCreate(&b));invoke(false);invoke(true);CK(cudaDeviceSynchronize());
    for(int trial=0;trial<5;++trial)for(int k=0;k<2;++k){bool p=((k+trial)%2)!=0;int repeats=p?20:1;CK(cudaEventRecord(a));for(int i=0;i<repeats;++i)invoke(p);CK(cudaEventRecord(b));CK(cudaEventSynchronize(b));float ms;CK(cudaEventElapsedTime(&ms,a,b));printf("case=%d rows=%d hidden=%d trial=%d parallel=%d chain_ms=%.9f\n",id,r,h,trial,p,ms/repeats);}
    CK(cudaEventDestroy(a));CK(cudaEventDestroy(b));
  }
  // Independent zero-input identity: no projection/reference kernel is involved.
  CK(cudaMemset(x,0,r*width*2));CK(cudaMemset(base,0,controls*4));invoke(true);CK(cudaDeviceSynchronize());
  for(float v:read(post,r*s))if(v!=1.f)exit(4);
  for(auto v:read(y,r*h))if(__bfloat162float(v)!=0.f)exit(5);
  for(float v:read(matrix,r*s*s))if(v!=(1.f/s+1e-6f))exit(6);
  CK(cudaFree(counts));CK(cudaFree(x));CK(cudaFree(w));CK(cudaFree(base));CK(cudaFree(scale));CK(cudaFree(scratch));CK(cudaFree(post));CK(cudaFree(matrix));CK(cudaFree(y));
  printf("case=%d correctness=bitwise identity=passed released=true\n",id);
}
int main(int argc,char**){bool sanitize=argc>1;run(0,2,17,3,launch0,sanitize);run(1,4,129,4,launch1,sanitize);run(2,4,4096,16,launch2,sanitize);run(3,16,7,2,launch3,sanitize);CK(cudaDeviceReset());puts("all_cases=passed context_closed=true");}
