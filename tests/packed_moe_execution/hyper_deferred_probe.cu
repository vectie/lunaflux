#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "probe.cuh"
#define CK(x) do{auto e=(x);if(e!=cudaSuccess){fprintf(stderr,"%s: %s\n",#x,cudaGetErrorString(e));exit(2);}}while(0)
using Launch=void(*)(int,int*,float*,float*,__nv_bfloat16*,__nv_bfloat16*,__nv_bfloat16*,__nv_bfloat16*,float*,cudaStream_t);
template<class T>T* alloc(size_t n){T*p;CK(cudaMalloc(&p,n*sizeof(T)));return p;}
template<class T>void put(T*p,const std::vector<T>&v){CK(cudaMemcpy(p,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice));}
template<class T>std::vector<T> get(T*p,size_t n){std::vector<T>v(n);CK(cudaMemcpy(v.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost));return v;}
template<class T>void equal(const std::vector<T>&a,const std::vector<T>&b){if(a.size()!=b.size()||memcmp(a.data(),b.data(),a.size()*sizeof(T))){for(size_t i=0;i<a.size();++i)if(memcmp(&a[i],&b[i],sizeof(T))){fprintf(stderr,"bitwise mismatch index=%zu\n",i);break;}exit(3);}}
void run(int id,int s,int h,int r,Launch launch,bool sanitizer){
 const int controls=s*(s+2), scratch_count=r*(1+controls);
 auto counts=alloc<int>(5);auto post=alloc<float>(r*s);auto comb=alloc<float>(r*s*s);
 auto x=alloc<__nv_bfloat16>(r*s*h);auto b=alloc<__nv_bfloat16>(r*h);auto y=alloc<__nv_bfloat16>(r*s*h);
 auto weight=alloc<__nv_bfloat16>(controls*s*h);auto scratch=alloc<float>(scratch_count);
 std::vector<__nv_bfloat16> hx(r*s*h),hb(r*h),hw(controls*s*h);std::vector<float> hp(r*s),hc(r*s*s);
 for(size_t i=0;i<hx.size();++i)hx[i]=__float2bfloat16_rn((int(i*37%211)-105)*.03125f);
 for(size_t i=0;i<hb.size();++i)hb[i]=__float2bfloat16_rn((int(i*17%101)-50)*.015625f);
 for(size_t i=0;i<hw.size();++i)hw[i]=__float2bfloat16_rn((int(i*13%127)-63)*.00390625f);
 for(size_t i=0;i<hp.size();++i)hp[i]=.7f+float(i%7)*.013f;
 for(size_t i=0;i<hc.size();++i)hc[i]=.1f+float(i%11)*.017f;
 put(x,hx);put(b,hb);put(weight,hw);put(post,hp);put(comb,hc);
 cudaStream_t stream;CK(cudaStreamCreate(&stream));
 auto invoke=[&](int v){launch(v,counts,post,comb,x,b,y,weight,scratch,stream);CK(cudaGetLastError());};
 auto clear=[&](){CK(cudaMemset(y,0x5a,r*s*h*2));CK(cudaMemset(scratch,0x5a,scratch_count*4));};
 for(int live: {0,1,r-1,r,r+1}){
  put(counts,std::vector<int>{1,0,1,live,0});clear();invoke(0);CK(cudaDeviceSynchronize());auto oy=get(y,r*s*h);auto os=get(scratch,scratch_count);
  for(int variant:{1,2,4,8}){clear();invoke(variant);CK(cudaDeviceSynchronize());equal(oy,get(y,r*s*h));equal(os,get(scratch,scratch_count));}
 }
 if(!sanitizer){
  for(int live:{1,22,36,64}){if(live>r)continue;put(counts,std::vector<int>{1,0,1,live,0});
   cudaGraphExec_t graphs[2];const int variants[]={0,8};
   for(int k=0;k<2;++k){invoke(variants[k]);CK(cudaDeviceSynchronize());
    CK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));
    for(int i=0;i<100;++i)invoke(variants[k]);
    cudaGraph_t graph;CK(cudaStreamEndCapture(stream,&graph));CK(cudaGraphInstantiate(&graphs[k],graph,0));CK(cudaGraphDestroy(graph));
   }
   cudaEvent_t a,z;CK(cudaEventCreate(&a));CK(cudaEventCreate(&z));
   for(int trial=0;trial<7;++trial)for(int k=0;k<2;++k){int selected=(trial+k)%2;CK(cudaEventRecord(a,stream));CK(cudaGraphLaunch(graphs[selected],stream));CK(cudaEventRecord(z,stream));CK(cudaEventSynchronize(z));float ms;CK(cudaEventElapsedTime(&ms,a,z));printf("case=%d live=%d trial=%d variant=%d ms=%.9f\n",id,live,trial,variants[selected],ms/100);}
   CK(cudaEventDestroy(a));CK(cudaEventDestroy(z));for(auto graph:graphs)CK(cudaGraphExecDestroy(graph));
  }
 }
 CK(cudaStreamDestroy(stream));CK(cudaFree(counts));CK(cudaFree(post));CK(cudaFree(comb));CK(cudaFree(x));CK(cudaFree(b));CK(cudaFree(y));CK(cudaFree(weight));CK(cudaFree(scratch));printf("case=%d publication_inverse_projection=bitwise released=true\n",id);
}
int main(int argc,char**){run(0,4,64,3,launch0,argc>1);run(1,4,320,4,launch1,argc>1);run(2,4,4096,64,launch2,argc>1);run(3,16,64,2,launch3,argc>1);CK(cudaDeviceReset());puts("all_passed context_closed=true");}
