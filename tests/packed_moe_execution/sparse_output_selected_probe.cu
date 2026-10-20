#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <mma.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "selected.cuh"
#define CK(e) do { auto s=(e); if(s!=cudaSuccess){fprintf(stderr,"%s\n",cudaGetErrorString(s));exit(2);} } while(0)
#define CB(e) do { if((e)!=CUBLAS_STATUS_SUCCESS) exit(3); } while(0)
template<class T> T* alloc(size_t n){T*p;CK(cudaMalloc(&p,n*sizeof(T)));return p;}
int main(int argc,char **argv) {
 const int m=64,n=4096,k=8192;
 auto x=alloc<__nv_bfloat16>(m*k),w=alloc<__nv_bfloat16>(size_t(n)*k);
 auto y=alloc<__nv_bfloat16>(m*n),z=alloc<__nv_bfloat16>(m*n);
 auto counts=alloc<int>(4);
 std::vector<__nv_bfloat16> hx(m*k),hw(size_t(n)*k),a(m*n),b(m*n);
 for(size_t i=0;i<hx.size();++i)hx[i]=__float2bfloat16_rn((int(i*37%257)-128)*.015625f);
 for(size_t i=0;i<hw.size();++i)hw[i]=__float2bfloat16_rn((int(i*53%251)-125)*.00390625f);
 CK(cudaMemcpy(x,hx.data(),hx.size()*2,cudaMemcpyHostToDevice));
 CK(cudaMemcpy(w,hw.data(),hw.size()*2,cudaMemcpyHostToDevice));
 int c[]={1,0,1,m};CK(cudaMemcpy(counts,c,sizeof(c),cudaMemcpyHostToDevice));
 cudaStream_t stream;CK(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
 cublasHandle_t handle;CB(cublasCreate(&handle));CB(cublasSetStream(handle,stream));float alpha=1,beta=0;
 auto launch=[&](int variant) {
  if(variant==0)selected_before<<<1024,128,0,stream>>>(counts,x,w,y);
  else if(variant==1)selected_after<<<1024,128,0,stream>>>(counts,x,w,z);
  else CB(cublasGemmEx(handle,CUBLAS_OP_T,CUBLAS_OP_N,n,m,k,&alpha,w,CUDA_R_16BF,k,x,CUDA_R_16BF,k,&beta,z,CUDA_R_16BF,n,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  CK(cudaGetLastError());
 };
 if(argc>1) {launch(atoi(argv[1]));CK(cudaDeviceSynchronize());}
 else {
  launch(0);launch(1);CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(a.data(),y,a.size()*2,cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(b.data(),z,b.size()*2,cudaMemcpyDeviceToHost));
  if(memcmp(a.data(),b.data(),a.size()*2))return 4;
  puts("actual_selected_bf16_outputs=bitwise; vendor_is_not_correctness_oracle");
  cudaGraphExec_t graphs[3];
  for(int v=0;v<3;++v){
   launch(v);CK(cudaDeviceSynchronize());
   CK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));
   for(int r=0;r<20;++r)launch(v);
   cudaGraph_t graph;CK(cudaStreamEndCapture(stream,&graph));CK(cudaGraphInstantiate(&graphs[v],graph,0));CK(cudaGraphDestroy(graph));
  }
  cudaEvent_t start,end;CK(cudaEventCreate(&start));CK(cudaEventCreate(&end));
  for(int trial=0;trial<5;++trial)for(int p=0;p<3;++p){
   int v=(trial+p)%3;CK(cudaEventRecord(start,stream));CK(cudaGraphLaunch(graphs[v],stream));CK(cudaEventRecord(end,stream));CK(cudaEventSynchronize(end));
   float ms;CK(cudaEventElapsedTime(&ms,start,end));printf("trial=%d variant=%d ms=%.9f\n",trial,v,ms/20);
  }
  for(auto g:graphs)CK(cudaGraphExecDestroy(g));CK(cudaEventDestroy(start));CK(cudaEventDestroy(end));
 }
 CB(cublasDestroy(handle));CK(cudaStreamDestroy(stream));for(void*p:{(void*)x,(void*)w,(void*)y,(void*)z,(void*)counts})CK(cudaFree(p));CK(cudaDeviceReset());puts("released=true");
}
