#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "index.cuh"
static void ck(cudaError_t e){if(e!=cudaSuccess){fprintf(stderr,"%s\n",cudaGetErrorString(e));exit(1);}}
template<class T>static T* alloc(size_t n){T*p;ck(cudaMalloc(&p,n*sizeof(T)));return p;}
int main(int argc,char**argv){
 const bool sanitize=argc>1; const int R=16,H=256,C=64,W=259;
 auto counts=alloc<int>(5),pos=alloc<int>(R),raw=alloc<int>(H),valid=alloc<int>(C),n=alloc<int>(1),keyvalid=alloc<int>(H),a=alloc<int>(R*W),b=alloc<int>(R*W);
 auto q=alloc<__nv_bfloat16>(R*128),hw=alloc<__nv_bfloat16>(R*4),k=alloc<__nv_bfloat16>(C*32);auto scratch=alloc<unsigned char>(scratch_bytes);
 std::vector<__nv_bfloat16>hq(R*128),hweights(R*4),hk(C*32);std::vector<int>hp(R),hr(H),hv(C,1),hkv(H,1),out(R*W),ref(R*W);
 for(int i=0;i<R*128;++i)hq[i]=__float2bfloat16_rn((i%11-5)*.125f);
 for(int i=0;i<R*4;++i)hweights[i]=__float2bfloat16_rn((i%7-3)*.25f);
 for(int i=0;i<C*32;++i)hk[i]=__float2bfloat16_rn((i%17-8)*.125f);
 for(int i=0;i<H;++i)hr[i]=i;
 ck(cudaMemcpy(q,hq.data(),hq.size()*2,cudaMemcpyHostToDevice));ck(cudaMemcpy(hw,hweights.data(),hweights.size()*2,cudaMemcpyHostToDevice));ck(cudaMemcpy(k,hk.data(),hk.size()*2,cudaMemcpyHostToDevice));ck(cudaMemcpy(raw,hr.data(),H*4,cudaMemcpyHostToDevice));
 auto run=[&](bool retained){if(retained)bench_index<<<R,128>>>(counts,pos,q,hw,k,raw,valid,n,keyvalid,b,(float*)scratch);else serial_index<<<1,1>>>(counts,pos,q,hw,k,raw,valid,n,keyvalid,a);ck(cudaGetLastError());};
 for(int scenario=0;scenario<10;++scenario){
  int hc[]={0,1,1,scenario==0?0:scenario==1?17:scenario==2?1:16,0};int cn=scenario==3?65:scenario==4?0:C;
  for(int r=0;r<R;++r)hp[r]=scenario==5?-1:scenario==6?256:scenario==7?r:255-r;
  for(int i=0;i<C;++i)hv[i]=(scenario==8&&i%3==0)?0:1;
  for(int i=0;i<H;++i)hkv[i]=(scenario==9&&i==4)?0:1;
  ck(cudaMemcpy(counts,hc,20,cudaMemcpyHostToDevice));ck(cudaMemcpy(n,&cn,4,cudaMemcpyHostToDevice));ck(cudaMemcpy(pos,hp.data(),R*4,cudaMemcpyHostToDevice));ck(cudaMemcpy(valid,hv.data(),C*4,cudaMemcpyHostToDevice));ck(cudaMemcpy(keyvalid,hkv.data(),H*4,cudaMemcpyHostToDevice));
  run(false);run(true);ck(cudaMemcpy(ref.data(),a,R*W*4,cudaMemcpyDeviceToHost));ck(cudaMemcpy(out.data(),b,R*W*4,cudaMemcpyDeviceToHost));
  if(ref!=out){fprintf(stderr,"indices differ scenario=%d\n",scenario);return 3;}
 }
 // Analytic all-zero scores require stable ascending candidate ties.
 ck(cudaMemset(hw,0,R*4*2));for(int r=0;r<R;++r)hp[r]=255;for(int i=0;i<H;++i)hkv[i]=1;
 ck(cudaMemcpy(pos,hp.data(),R*4,cudaMemcpyHostToDevice));ck(cudaMemcpy(keyvalid,hkv.data(),H*4,cudaMemcpyHostToDevice));run(true);ck(cudaMemcpy(out.data(),b,R*W*4,cudaMemcpyDeviceToHost));
 for(int r=0;r<R;++r)for(int i=0;i<W;++i)if(out[r*W+i]!=(i<256?i:-1)){fprintf(stderr,"stable tie oracle failed\n");return 4;}
 if(!sanitize)for(int trial=0;trial<5;++trial)for(bool retained:{false,true}){
  cudaEvent_t begin,end;ck(cudaEventCreate(&begin));ck(cudaEventCreate(&end));ck(cudaEventRecord(begin));for(int i=0;i<5;++i)run(retained);ck(cudaEventRecord(end));ck(cudaEventSynchronize(end));float ms;ck(cudaEventElapsedTime(&ms,begin,end));printf("trial=%d retained=%d milliseconds=%.9f\n",trial,retained,ms/5);ck(cudaEventDestroy(begin));ck(cudaEventDestroy(end));
 }
 for(void*p:{(void*)counts,(void*)pos,(void*)raw,(void*)valid,(void*)n,(void*)keyvalid,(void*)a,(void*)b,(void*)q,(void*)hw,(void*)k,(void*)scratch})ck(cudaFree(p));ck(cudaDeviceReset());puts("indices_bitwise=true stable_tie_oracle=pass released=true");
}
