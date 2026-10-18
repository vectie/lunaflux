#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <limits>
#include "probe.cuh"
#define CK(x) do{auto e=(x);if(e!=cudaSuccess){fprintf(stderr,"%s: %s\n",#x,cudaGetErrorString(e));exit(2);}}while(0)
using Launch=void(*)(bool,int*,__nv_bfloat16*,float*,float*,float*,float*,float*,float*,__nv_bfloat16*,__nv_bfloat16*,__nv_bfloat16*,cudaStream_t);
template<class T>T* alloc(size_t n){T*p;CK(cudaMalloc(&p,n*sizeof(T)));return p;}
template<class T>void put(T*p,const std::vector<T>&v){CK(cudaMemcpy(p,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice));}
template<class T>std::vector<T> get(T*p,size_t n){std::vector<T>v(n);CK(cudaMemcpy(v.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost));return v;}
template<class T>void equal(const std::vector<T>&a,const std::vector<T>&b){if(a.size()!=b.size()||memcmp(a.data(),b.data(),a.size()*sizeof(T))){fprintf(stderr,"bitwise mismatch\n");exit(3);}}
void run(int id,int s,int h,int r,Launch launch,bool sanitizer){
 int controls=s*(s+2);auto counts=alloc<int>(5);auto x=alloc<__nv_bfloat16>(r*s*h);auto scratch=alloc<float>(r*(controls+1));auto base=alloc<float>(controls);auto scale=alloc<float>(3);auto post=alloc<float>(r*s);auto initial=alloc<float>(r*s*s);auto combine=alloc<float>(r*s*s);auto y=alloc<__nv_bfloat16>(r*h);auto w=alloc<__nv_bfloat16>(h);auto norm=alloc<__nv_bfloat16>(r*h);
 std::vector<__nv_bfloat16> hx(r*s*h);for(size_t i=0;i<hx.size();++i)hx[i]=__float2bfloat16_rn((int(i*37%211)-105)*.03125f);put(x,hx);put(w,std::vector<__nv_bfloat16>(h,__float2bfloat16_rn(1.0f)));
 std::vector<float> hs(r*(controls+1)),hb(controls);for(size_t i=0;i<hs.size();++i)hs[i]=i<size_t(r)?1.0f:(int(i*17%101)-50)*.03125f;for(int i=0;i<controls;++i)hb[i]=(i-5)*.03125f;put(scratch,hs);put(base,hb);put(scale,std::vector<float>{.125f,.25f,.125f});
 auto clear=[&](){CK(cudaMemset(post,0x5a,r*s*4));CK(cudaMemset(initial,0x5a,r*s*s*4));CK(cudaMemset(combine,0x5a,r*s*s*4));CK(cudaMemset(y,0x5a,r*h*2));CK(cudaMemset(norm,0x5a,r*h*2));};
 cudaStream_t stream;CK(cudaStreamCreate(&stream));
 auto invoke=[&](bool fused){launch(fused,counts,x,scratch,base,scale,post,initial,combine,y,w,norm,stream);CK(cudaGetLastError());};
 auto check=[&](){clear();invoke(false);CK(cudaDeviceSynchronize());auto op=get(post,r*s),oi=get(initial,r*s*s),oc=get(combine,r*s*s);auto oy=get(y,r*h),on=get(norm,r*h);clear();invoke(true);CK(cudaDeviceSynchronize());equal(op,get(post,r*s));equal(oi,get(initial,r*s*s));equal(oc,get(combine,r*s*s));equal(oy,get(y,r*h));equal(on,get(norm,r*h));};
 for(int live:{0,1,r-1,r,r+1}){put(counts,std::vector<int>{1,0,1,live,0});check();}
 put(counts,std::vector<int>{1,0,1,r,0});
 auto bad_base=hb;bad_base[controls-1]=std::numeric_limits<float>::quiet_NaN();put(base,bad_base);check();put(base,hb);
 put(scale,std::vector<float>{.125f,std::numeric_limits<float>::infinity(),.125f});check();put(scale,std::vector<float>{.125f,.25f,.125f});
 auto extreme=hs;for(size_t i=r;i<extreme.size();++i)extreme[i]=(i%2?1024.0f:-1024.0f);put(scratch,extreme);check();put(scratch,hs);
 if(!sanitizer){cudaEvent_t a,z;CK(cudaEventCreate(&a));CK(cudaEventCreate(&z));for(int live:{1,22,36,64}){if(live>r)continue;put(counts,std::vector<int>{1,0,1,live,0});invoke(false);invoke(true);CK(cudaDeviceSynchronize());
  cudaGraphExec_t graphs[2];for(int k=0;k<2;++k){CK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));for(int i=0;i<100;++i)invoke(k);cudaGraph_t graph;CK(cudaStreamEndCapture(stream,&graph));CK(cudaGraphInstantiate(&graphs[k],graph,0));CK(cudaGraphDestroy(graph));CK(cudaGraphLaunch(graphs[k],stream));CK(cudaStreamSynchronize(stream));}
  for(int trial=0;trial<7;++trial)for(int k=0;k<2;++k){int fused=(trial+k)%2;CK(cudaEventRecord(a,stream));CK(cudaGraphLaunch(graphs[fused],stream));CK(cudaEventRecord(z,stream));CK(cudaEventSynchronize(z));float ms;CK(cudaEventElapsedTime(&ms,a,z));printf("case=%d live=%d trial=%d fused=%d ms=%.9f\n",id,live,trial,fused,ms/100);}for(auto graph:graphs)CK(cudaGraphExecDestroy(graph));}CK(cudaEventDestroy(a));CK(cudaEventDestroy(z));}
 CK(cudaStreamDestroy(stream));CK(cudaFree(counts));CK(cudaFree(x));CK(cudaFree(scratch));CK(cudaFree(base));CK(cudaFree(scale));CK(cudaFree(post));CK(cudaFree(initial));CK(cudaFree(combine));CK(cudaFree(y));CK(cudaFree(w));CK(cudaFree(norm));printf("case=%d all_five_frontiers=bitwise released=true\n",id);
}
int main(int argc,char**){run(0,4,64,3,launch0,argc>1);run(1,4,320,4,launch1,argc>1);run(2,4,4096,64,launch2,argc>1);run(3,16,64,2,launch3,argc>1);CK(cudaDeviceReset());puts("context_closed=true");}
