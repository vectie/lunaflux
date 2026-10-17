// Offline actual-source probe. No production dependency on cuBLAS or counters.
#include "projection.cuh"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
static void ck(cudaError_t e){if(e!=cudaSuccess){fprintf(stderr,"%s\n",cudaGetErrorString(e));exit(1);}}
static void cb(cublasStatus_t e){if(e!=CUBLAS_STATUS_SUCCESS){fprintf(stderr,"cublas=%d\n",int(e));exit(2);}}
template<class T>T* alloc(size_t n){T*p;ck(cudaMalloc(&p,n*sizeof(T)));return p;}
int main(int argc,char**argv){
 const int k=4096,n=12576; const char*mode=argc>1?argv[1]:"timing";
 std::vector<__nv_bfloat16>x(k),w(size_t(n)*k),h(n),reference(n);
 unsigned seed=2391;auto sample=[&](){seed=1664525U*seed+1013904223U;return float(int(seed>>8)%20001-10000)/13001.0f;};
 for(auto&v:x)v=__float2bfloat16_rn(sample());for(auto&v:w)v=__float2bfloat16_rn(sample());
 auto dx=alloc<__nv_bfloat16>(k),dw=alloc<__nv_bfloat16>(w.size()),y=alloc<__nv_bfloat16>(n),v=alloc<__nv_bfloat16>(n);
 auto counts=alloc<int>(4);int c[]={0,1,1,1};
 ck(cudaMemcpy(dx,x.data(),k*2,cudaMemcpyHostToDevice));ck(cudaMemcpy(dw,w.data(),w.size()*2,cudaMemcpyHostToDevice));ck(cudaMemcpy(counts,c,sizeof(c),cudaMemcpyHostToDevice));
 cublasHandle_t handle;cb(cublasCreate(&handle));float alpha=1,beta=0;
 auto run=[&](int route){
 #define ARGS counts,dx,dw,y,dw+4096ULL*k,y+4096,dw+8192ULL*k,y+8192,dw+12288ULL*k,y+12288,dw+12416ULL*k,y+12416,dw+12448ULL*k,y+12448
 if(route==0)projection_old<<<3144,128>>>(ARGS);
 if(route==1)projection_p4<<<3144,128>>>(ARGS);
 if(route==2)projection_p8<<<3144,128>>>(ARGS);
 if(route==3)cb(cublasGemmEx(handle,CUBLAS_OP_T,CUBLAS_OP_N,n,1,k,&alpha,dw,CUDA_R_16BF,k,dx,CUDA_R_16BF,k,&beta,v,CUDA_R_16BF,n,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP));
 ck(cudaGetLastError());};
 if(!strcmp(mode,"profile")){for(int r=0;r<4;++r)run(r);}
 else {
 run(0);ck(cudaMemcpy(reference.data(),y,n*2,cudaMemcpyDeviceToHost));
 for(int r=1;r<3;++r){run(r);ck(cudaMemcpy(h.data(),y,n*2,cudaMemcpyDeviceToHost));if(memcmp(reference.data(),h.data(),n*2)){fprintf(stderr,"route=%d bitwise mismatch\n",r);return 4;}printf("route=%d bitwise=%d/%d\n",r,n,n);}
 run(3);ck(cudaMemcpy(h.data(),v,n*2,cudaMemcpyDeviceToHost));double e=0,s=0;for(int i=0;i<n;++i){double a=__bfloat162float(reference[i]),b=__bfloat162float(h[i]);e+=(a-b)*(a-b);s+=b*b;}printf("vendor_rel_l2=%.9g (comparison, not oracle)\n",sqrt(e/s));
 if(strcmp(mode,"sanitizer")){
 auto time=[&](int r){cudaEvent_t a,b;ck(cudaEventCreate(&a));ck(cudaEventCreate(&b));ck(cudaEventRecord(a));for(int i=0;i<20;++i)run(r);ck(cudaEventRecord(b));ck(cudaEventSynchronize(b));float ms;ck(cudaEventElapsedTime(&ms,a,b));ck(cudaEventDestroy(a));ck(cudaEventDestroy(b));return ms/20;};
 for(int r=0;r<4;++r)run(r);ck(cudaDeviceSynchronize());
 for(int t=0;t<6;++t){for(int j=0;j<4;++j){int r=t%2?3-j:j;printf("trial=%d route=%d ms=%.9g\n",t,r,time(r));}}
 }}
 ck(cudaDeviceSynchronize());cb(cublasDestroy(handle));for(void*p:{(void*)dx,(void*)dw,(void*)y,(void*)v,(void*)counts})ck(cudaFree(p));ck(cudaDeviceReset());puts("released=true");
}
