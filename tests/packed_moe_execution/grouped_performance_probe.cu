#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cmath>
#include "grouped.cuh"
static void ck(cudaError_t e){if(e!=cudaSuccess){fprintf(stderr,"%s\n",cudaGetErrorString(e));exit(1);}}
template<class T>static T* alloc(size_t n){T*p;ck(cudaMalloc(&p,n*sizeof(T)));return p;}
int main(int argc,char**argv){
 const int H=4096,I=2048,R=16,S=8,J=R*S,E=32;
 const int rows=argc>1?atoi(argv[1]):16;
 const char*mode=argc>2?argv[2]:"timing";
 if(rows!=1&&rows!=16)return 2;
 auto bank=alloc<unsigned char>(matrix_stride*E);
 for(int e=0;e<E;++e)for(int stage=0;stage<3;++stage){
   size_t elements=H*I;auto p=bank+e*matrix_stride+matrix_offsets[stage];
   ck(cudaMemset(p,0x22,elements/2));ck(cudaMemset(p+elements/2,0x38,elements/16));
   float scale=.03125f;ck(cudaMemcpy(p+elements/2+elements/16,&scale,4,cudaMemcpyHostToDevice));
 }
 auto x=alloc<__nv_bfloat16>(R*H);std::vector<__nv_bfloat16>hx(R*H);
 for(int i=0;i<R*H;++i)hx[i]=__float2bfloat16_rn((i%13-6)*.03125f);
 ck(cudaMemcpy(x,hx.data(),R*H*2,cudaMemcpyHostToDevice));
 auto counts=alloc<int>(5),indices=alloc<int>(J),map=alloc<int>(288);auto weights=alloc<float>(J);
 int hc[]={0,1,1,rows,0};std::vector<int>hi(J),hm(288,-1);std::vector<float>hw(J,.125f);
 for(int i=0;i<E;++i)hm[i]=i;for(int r=0;r<R;++r)for(int s=0;s<S;++s)hi[r*S+s]=(r*3+s)%E;
 ck(cudaMemcpy(counts,hc,20,cudaMemcpyHostToDevice));ck(cudaMemcpy(indices,hi.data(),J*4,cudaMemcpyHostToDevice));ck(cudaMemcpy(map,hm.data(),288*4,cudaMemcpyHostToDevice));ck(cudaMemcpy(weights,hw.data(),J*4,cudaMemcpyHostToDevice));
 auto a=alloc<unsigned char>(matrix_activation);auto b=alloc<__nv_bfloat16>(J*H);auto y=alloc<float>(R*H);
 auto ab=(__nv_bfloat16*)a;
 auto run=[&](bool matrix){
  if(matrix){matrix_group<<<1,128>>>(counts,indices,ab);matrix_gate_up<<<4096,128>>>(counts,x,indices,map,bank,ab);matrix_down<<<8192,128>>>(counts,ab,indices,weights,map,bank,b);matrix_combine<<<256,256>>>(counts,indices,b,y);}
  else{scalar_gate_up<<<65536,128>>>(counts,x,indices,map,bank,ab);scalar_down<<<131072,128>>>(counts,ab,indices,weights,map,bank,b);scalar_combine<<<256,256>>>(counts,indices,b,y);}
  ck(cudaGetLastError());
 };
 if(!strcmp(mode,"matrix")||!strcmp(mode,"scalar")){run(!strcmp(mode,"matrix"));ck(cudaDeviceSynchronize());}
 else{
  std::vector<float>ref(rows*H),actual(rows*H);
  run(false);ck(cudaMemcpy(ref.data(),y,rows*H*4,cudaMemcpyDeviceToHost));
  run(true);ck(cudaMemcpy(actual.data(),y,rows*H*4,cudaMemcpyDeviceToHost));
  if(memcmp(ref.data(),actual.data(),rows*H*4)){fprintf(stderr,"dyadic output mismatch\n");return 3;}
  for(bool matrix:{false,true})for(int trial=0;trial<5;++trial){
   cudaEvent_t begin,end;ck(cudaEventCreate(&begin));ck(cudaEventCreate(&end));
   ck(cudaEventRecord(begin));for(int i=0;i<5;++i)run(matrix);ck(cudaEventRecord(end));ck(cudaEventSynchronize(end));float ms;ck(cudaEventElapsedTime(&ms,begin,end));
   printf("rows=%d route=%s trial=%d milliseconds=%.9f\n",rows,matrix?"matrix":"scalar",trial,ms/5);
   ck(cudaEventDestroy(begin));ck(cudaEventDestroy(end));
  }
 }
 for(void*p:{(void*)bank,(void*)x,(void*)counts,(void*)indices,(void*)map,(void*)weights,(void*)a,(void*)b,(void*)y})ck(cudaFree(p));
 ck(cudaDeviceReset());puts("released=true");
}
