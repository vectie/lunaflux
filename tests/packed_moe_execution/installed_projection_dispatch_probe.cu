// Offline replay of the installed reference's GEMM dispatch. Not linked into
// LunaFlux. Mathematical comparison is separate from dispatch identification.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <cublasLt.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#define CK(e) do {auto s=(e);if(s!=cudaSuccess){fprintf(stderr,"%s\n",cudaGetErrorString(s));exit(2);}}while(0)
#define CB(e) do {auto s=(e);if(s!=CUBLAS_STATUS_SUCCESS){fprintf(stderr,"cublas=%d\n",int(s));exit(3);}}while(0)
template<class T>T* alloc(size_t n){T*p;CK(cudaMalloc(&p,n*sizeof(T)));return p;}
int main(int argc,char**argv) {
 if(argc!=2)return 2;
 const int route=atoi(argv[1]);if(route<0||route>9)return 2;
 const int m=64,n=4096,k=8192;
 auto x=alloc<__nv_bfloat16>(m*k),w=alloc<__nv_bfloat16>(size_t(n)*k),y=alloc<__nv_bfloat16>(m*n);
 auto workspace=alloc<char>(32*1024*1024);
 // Same literal deterministic operands as the retained projection trial.
 // Zero-filled operands are sufficient for dispatch identification, but not
 // for a matched data-dependent memory/progress comparison.
 std::vector<__nv_bfloat16> hx(m*k),hw(size_t(n)*k);
 unsigned seed=713;
 for(auto*values:{&hx,&hw})for(auto&v:*values){
  seed=1664525u*seed+1013904223u;
  v=__float2bfloat16_rn(float(int(seed>>16)-32768)*.000001f);
 }
 CK(cudaMemcpy(x,hx.data(),hx.size()*2,cudaMemcpyHostToDevice));
 CK(cudaMemcpy(w,hw.data(),hw.size()*2,cudaMemcpyHostToDevice));
 cublasHandle_t h;CB(cublasCreate(&h));int version=0;CB(cublasGetVersion(h,&version));
 CB(cublasSetWorkspace(h,workspace,32*1024*1024));
 printf("version=%d route=%d M=%d N=%d K=%d workspace=33554432 operands=projection_trial_seed713\n",version,route,m,n,k);
 int result=0;
 const float alpha=1,beta=0;
 if(route<2) {
  CB(cublasGemmEx(h,CUBLAS_OP_T,CUBLAS_OP_N,n,m,k,&alpha,w,CUDA_R_16BF,k,x,CUDA_R_16BF,k,&beta,y,CUDA_R_16BF,n,CUBLAS_COMPUTE_32F,route==0?CUBLAS_GEMM_DEFAULT:CUBLAS_GEMM_DEFAULT_TENSOR_OP));
 } else {
  cublasLtHandle_t lt;CB(cublasLtCreate(&lt));
  cublasLtMatmulDesc_t op;CB(cublasLtMatmulDescCreate(&op,CUBLAS_COMPUTE_32F,CUDA_R_32F));
  cublasOperation_t transpose=CUBLAS_OP_T;
  CB(cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_TRANSA,&transpose,sizeof(transpose)));
  cublasLtMatrixLayout_t a,b,c;
  CB(cublasLtMatrixLayoutCreate(&a,CUDA_R_16BF,k,n,k));
  CB(cublasLtMatrixLayoutCreate(&b,CUDA_R_16BF,k,m,k));
  CB(cublasLtMatrixLayoutCreate(&c,CUDA_R_16BF,n,m,n));
  cublasLtMatmulPreference_t pref;CB(cublasLtMatmulPreferenceCreate(&pref));
  size_t bytes=32*1024*1024;
  CB(cublasLtMatmulPreferenceSetAttribute(pref,CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&bytes,sizeof(bytes)));
  cublasLtMatmulHeuristicResult_t results[8];int found=0;
  CB(cublasLtMatmulAlgoGetHeuristic(lt,op,a,b,c,c,pref,8,results,&found));
  int index=route-2;
  if(index>=found){fprintf(stderr,"missing heuristic %d/%d\n",index,found);result=4;}
  else {
   printf("heuristic=%d count=%d workspace=%zu waves=%f\n",index,found,results[index].workspaceSize,results[index].wavesCount);
   CB(cublasLtMatmul(lt,op,&alpha,w,a,x,b,&beta,y,c,y,c,&results[index].algo,workspace,bytes,0));
  }
  CK(cudaDeviceSynchronize());
  CB(cublasLtMatmulPreferenceDestroy(pref));CB(cublasLtMatrixLayoutDestroy(a));
  CB(cublasLtMatrixLayoutDestroy(b));CB(cublasLtMatrixLayoutDestroy(c));
  CB(cublasLtMatmulDescDestroy(op));CB(cublasLtDestroy(lt));
 }
 CK(cudaDeviceSynchronize());CB(cublasDestroy(h));
 for(void*p:{(void*)x,(void*)w,(void*)y,(void*)workspace})CK(cudaFree(p));
 CK(cudaDeviceReset());puts("released=true");return result;
}
