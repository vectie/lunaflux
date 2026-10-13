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
 const int rows=argc>1?atoi(argv[1]):16;const char*mode=argc>2?argv[2]:"timing";
 if(rows!=1&&rows!=16)return 3;
 const int k=4096,n=8192;
 std::vector<__nv_bfloat16>x(rows*k),w(size_t(n)*k);
 for(size_t i=0;i<x.size();++i)x[i]=__float2bfloat16((int(i%31)-15)*.03125f);
 for(size_t i=0;i<w.size();++i)w[i]=__float2bfloat16((int(i%37)-18)*.015625f);
 auto dx=alloc<__nv_bfloat16>(x.size()),dw=alloc<__nv_bfloat16>(w.size()),y=alloc<__nv_bfloat16>(rows*n),ref=alloc<__nv_bfloat16>(rows*n);
 auto counts=alloc<int>(4);int c[]={0,1,1,rows};
 ck(cudaMemcpy(dx,x.data(),x.size()*2,cudaMemcpyHostToDevice));ck(cudaMemcpy(dw,w.data(),w.size()*2,cudaMemcpyHostToDevice));ck(cudaMemcpy(counts,c,sizeof(c),cudaMemcpyHostToDevice));
 cublasHandle_t handle;cb(cublasCreate(&handle));float alpha=1,beta=0;
 auto run=[&](bool vendor){if(vendor){cb(cublasGemmEx(handle,CUBLAS_OP_T,CUBLAS_OP_N,n,rows,k,&alpha,dw,CUDA_R_16BF,k,dx,CUDA_R_16BF,k,&beta,ref,CUDA_R_16BF,n,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP));}
 else glm_recurrent_25_0<<<512,256>>>(counts,dx,dw,y);ck(cudaGetLastError());};
 if(!strcmp(mode,"native"))run(false);else if(!strcmp(mode,"vendor"))run(true);
 else {
  run(false);run(true);ck(cudaDeviceSynchronize());std::vector<__nv_bfloat16>a(rows*n),b(a.size());
  ck(cudaMemcpy(a.data(),y,a.size()*2,cudaMemcpyDeviceToHost));ck(cudaMemcpy(b.data(),ref,b.size()*2,cudaMemcpyDeviceToHost));
  double maximum=0;for(size_t i=0;i<a.size();++i){double d=std::fabs(double(__bfloat162float(a[i]))-__bfloat162float(b[i]));if(!std::isfinite(d))return 4;maximum=std::max(maximum,d);}
  // Dyadic diagnostic inputs are exact here. This is not model quality testing.
  printf("diagnostic_maxabs=%.9g\n",maximum);if(maximum!=0)return 5;
  auto time=[&](bool vendor){cudaEvent_t a,b;ck(cudaEventCreate(&a));ck(cudaEventCreate(&b));ck(cudaEventRecord(a));for(int r=0;r<10;++r)run(vendor);ck(cudaEventRecord(b));ck(cudaEventSynchronize(b));float ms;ck(cudaEventElapsedTime(&ms,a,b));ck(cudaEventDestroy(a));ck(cudaEventDestroy(b));return ms/10;};
  for(int i=0;i<5;++i){float a,b;if(i%2){b=time(true);a=time(false);}else{a=time(false);b=time(true);}printf("rows=%d trial=%d native_ms=%.9g cublas_ms=%.9g\n",rows,i,a,b);}
 }
 ck(cudaDeviceSynchronize());cb(cublasDestroy(handle));for(void*p:{(void*)counts,(void*)ref,(void*)y,(void*)dw,(void*)dx})ck(cudaFree(p));ck(cudaDeviceReset());puts("released=true");
}
