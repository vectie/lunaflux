#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "probe.cuh"
#define CK(x) do{auto e=(x);if(e!=cudaSuccess){fprintf(stderr,"%s: %s\n",#x,cudaGetErrorString(e));exit(2);}}while(0)
using Launch=void(*)(bool,int*,float*,float*,__nv_bfloat16*,__nv_bfloat16*,__nv_bfloat16*,float*);
template<class T>T* alloc(size_t n){T*p;CK(cudaMalloc(&p,n*sizeof(T)));return p;}
template<class T>void put(T*p,const std::vector<T>&v){CK(cudaMemcpy(p,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice));}
template<class T>std::vector<T> get(T*p,size_t n){std::vector<T>v(n);CK(cudaMemcpy(v.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost));return v;}
template<class T>void equal(const std::vector<T>&a,const std::vector<T>&b){if(a.size()!=b.size()||memcmp(a.data(),b.data(),a.size()*sizeof(T))){fprintf(stderr,"bitwise mismatch\n");exit(3);}}
void run(int id,int s,int h,int r,Launch launch,bool sanitizer){
 auto counts=alloc<int>(5);auto post=alloc<float>(r*s);auto comb=alloc<float>(r*s*s);
 auto x=alloc<__nv_bfloat16>(r*s*h);auto b=alloc<__nv_bfloat16>(r*h);auto y=alloc<__nv_bfloat16>(r*s*h);auto scratch=alloc<float>(r*(1+s*(s+2)));
 std::vector<__nv_bfloat16> hx(r*s*h),hb(r*h);std::vector<float> hp(r*s),hc(r*s*s);
 for(size_t i=0;i<hx.size();++i)hx[i]=__float2bfloat16_rn((int(i*37%211)-105)*.03125f);
 for(size_t i=0;i<hb.size();++i)hb[i]=__float2bfloat16_rn((int(i*17%101)-50)*.015625f);
 for(size_t i=0;i<hp.size();++i)hp[i]=.7f+float(i%7)*.013f;
 for(size_t i=0;i<hc.size();++i)hc[i]=.1f+float(i%11)*.017f;
 put(x,hx);put(b,hb);put(post,hp);put(comb,hc);
 auto invoke=[&](bool fused){launch(fused,counts,post,comb,x,b,y,scratch);CK(cudaGetLastError());};
 for(int live: {0,1,r-1,r,r+1}){
  put(counts,std::vector<int>{1,0,1,live,0});
  CK(cudaMemset(y,0x5a,r*s*h*2));CK(cudaMemset(scratch,0x5a,r*4));invoke(false);CK(cudaDeviceSynchronize());auto oy=get(y,r*s*h);auto os=get(scratch,r);
  CK(cudaMemset(y,0x5a,r*s*h*2));CK(cudaMemset(scratch,0x5a,r*4));invoke(true);CK(cudaDeviceSynchronize());equal(oy,get(y,r*s*h));equal(os,get(scratch,r));
 }
 if(!sanitizer){cudaEvent_t a,z;CK(cudaEventCreate(&a));CK(cudaEventCreate(&z));
  for(int live:{1,22,36,64}){if(live>r)continue;put(counts,std::vector<int>{1,0,1,live,0});invoke(false);invoke(true);CK(cudaDeviceSynchronize());
   for(int trial=0;trial<5;++trial)for(int k=0;k<2;++k){bool fused=(trial+k)%2;CK(cudaEventRecord(a));for(int i=0;i<100;++i)invoke(fused);CK(cudaEventRecord(z));CK(cudaEventSynchronize(z));float ms;CK(cudaEventElapsedTime(&ms,a,z));printf("case=%d live=%d trial=%d fused=%d ms=%.9f\n",id,live,trial,fused,ms/100);}}
  CK(cudaEventDestroy(a));CK(cudaEventDestroy(z));}
 CK(cudaFree(counts));CK(cudaFree(post));CK(cudaFree(comb));CK(cudaFree(x));CK(cudaFree(b));CK(cudaFree(y));CK(cudaFree(scratch));printf("case=%d publication_and_inverse=bitwise released=true\n",id);
}
int main(int argc,char**){run(0,4,64,3,launch0,argc>1);run(1,4,320,4,launch1,argc>1);run(2,4,4096,64,launch2,argc>1);run(3,16,64,2,launch3,argc>1);CK(cudaDeviceReset());puts("all_passed context_closed=true");}
