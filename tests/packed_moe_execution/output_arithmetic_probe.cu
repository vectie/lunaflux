// Offline accumulation intervention on identical inverse-attention operands.
// Vendor arithmetic is a diagnostic reference only, never a serving dependency.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include "output_matched_kernels.cuh"
#include "output_striped_kernels.cuh"

static void check(cudaError_t e) {
  if(e!=cudaSuccess){std::fprintf(stderr,"%s\n",cudaGetErrorString(e));std::exit(2);}
}
static void blas(cublasStatus_t e) {
  if(e!=CUBLAS_STATUS_SUCCESS){std::fprintf(stderr,"cublas status %d\n",int(e));std::exit(2);}
}
template<class T> static T *load(const std::string &path,size_t n) {
  auto *file=std::fopen(path.c_str(),"rb");if(!file){std::perror(path.c_str());std::exit(3);}
  std::vector<T> values(n);
  const bool read=std::fread(values.data(),sizeof(T),n,file)==n&&std::fgetc(file)==EOF;
  std::fclose(file);if(!read){std::fprintf(stderr,"incomplete %s\n",path.c_str());std::exit(3);}
  T *p=nullptr;check(cudaMalloc(&p,n*sizeof(T)));
  check(cudaMemcpy(p,values.data(),n*sizeof(T),cudaMemcpyHostToDevice));return p;
}
template<class T> static T *allocate(size_t n) {
  T *p=nullptr;check(cudaMalloc(&p,n*sizeof(T)));check(cudaMemset(p,0,n*sizeof(T)));return p;
}
template<class T> static void save(T *p,size_t n,const std::string &path) {
  std::vector<T> values(n);check(cudaMemcpy(values.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost));
  auto *file=std::fopen(path.c_str(),"wbx");if(!file){std::perror(path.c_str());std::exit(4);}
  const bool wrote=std::fwrite(values.data(),sizeof(T),n,file)==n;
  const bool closed=std::fclose(file)==0;if(!wrote||!closed)std::exit(4);
}
__global__ void parameters(const unsigned char *w,const unsigned char *s,__nv_bfloat16 *out) {
  for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<8192*4096;i+=gridDim.x*blockDim.x) {
    out[i]=__float2bfloat16_rn(__fmul_rn(lf_block_e4m3_decode(w[i]),
      lf_block_e8m0_decode(s[(i/4096/128)*32+(i%4096/128)])));
  }
}
template<int N,bool Contiguous> __global__ void partials(
    const __nv_bfloat16 *x,const __nv_bfloat16 *w,__nv_bfloat16 *out) {
  const int row=blockIdx.x,group=blockIdx.y;
  for(int column=threadIdx.x;column<1024;column+=blockDim.x) {
    float sum[N]={};
    for(int i=0;i<4096;++i) {
      const int lane=Contiguous?i/(4096/N):i%N;
      sum[lane]=__fadd_rn(sum[lane],__fmul_rn(
        __bfloat162float(x[(row*8+group)*4096+i]),
        __bfloat162float(w[(group*1024+column)*4096+i])));
    }
    // Explicit tree: distinguish reassociation from multiplication contraction.
    for(int span=N/2;span>0;span/=2)
      for(int lane=0;lane<span;++lane)sum[lane]=__fadd_rn(sum[lane],sum[lane+span]);
    out[(row*8+group)*1024+column]=__float2bfloat16_rn(sum[0]);
  }
}
__global__ void accurate(const __nv_bfloat16 *x,const __nv_bfloat16 *w,__nv_bfloat16 *out) {
  const int row=blockIdx.x,group=blockIdx.y;
  for(int column=threadIdx.x;column<1024;column+=blockDim.x) {
    double sum=0.0;
    for(int i=0;i<4096;++i)sum+=double(__bfloat162float(x[(row*8+group)*4096+i]))*
      double(__bfloat162float(w[(group*1024+column)*4096+i]));
    out[(row*8+group)*1024+column]=__float2bfloat16_rn(float(sum));
  }
}
template<int N,bool C> static void run(const __nv_bfloat16 *x,const __nv_bfloat16 *w,
    __nv_bfloat16 *out,const std::string &root,const char *name) {
  partials<N,C><<<dim3(144,8),256>>>(x,w,out);
  check(cudaGetLastError());check(cudaDeviceSynchronize());save(out,144*8192,root+"/"+name+".bf16");
}
int main(int argc,char **argv) {
  if(argc<3||argc>5)return 2;
  const bool lowered_only=argc>=4&&std::string(argv[3])=="--lowered-only";
  const bool one_row=argc==5&&std::string(argv[4])=="--one-row";
  if((argc>=4&&!lowered_only)||(argc==5&&!one_row))return 2;
  const int live=one_row?1:144;
  const std::string root(argv[1]),out(argv[2]);
  auto *x=load<__nv_bfloat16>(root+"/input.bf16",144*32768);
  auto *wa=load<unsigned char>(root+"/wo_a.fp8",8192*4096);
  auto *sa=load<unsigned char>(root+"/wo_a.scale",64*32);
  auto *w=allocate<__nv_bfloat16>(8192*4096),*a=allocate<__nv_bfloat16>(144*8192);
  auto *counts=allocate<int>(5);int host[5]={0,0,0,live,0};
  check(cudaMemcpy(counts,host,sizeof(host),cudaMemcpyHostToDevice));
  lowered_output_grouped<<<dim3(live,1024),256>>>(counts,x,wa,sa,a);
  check(cudaGetLastError());check(cudaDeviceSynchronize());save(a,live*8192,out+"/lowered.bf16");
  host[3]=1;check(cudaMemcpy(counts,host,sizeof(host),cudaMemcpyHostToDevice));
  for(int row=0;row<live;++row)
    lowered_output_grouped<<<dim3(1,1024),256>>>(counts,x+row*32768,wa,sa,a+row*8192);
  check(cudaGetLastError());check(cudaDeviceSynchronize());save(a,live*8192,out+"/lowered_incremental.bf16");
  host[3]=144;check(cudaMemcpy(counts,host,sizeof(host),cudaMemcpyHostToDevice));
  if(!lowered_only) {
  matched_output_grouped<<<dim3(144,8),256>>>(counts,x,wa,sa,a);
  check(cudaGetLastError());check(cudaDeviceSynchronize());save(a,144*8192,out+"/production.bf16");
  parameters<<<256,256>>>(wa,sa,w);check(cudaGetLastError());check(cudaDeviceSynchronize());
  run<1,false>(x,w,a,out,"ordered");
  run<2,false>(x,w,a,out,"striped2");run<4,false>(x,w,a,out,"striped4");
  run<8,false>(x,w,a,out,"striped8");run<16,false>(x,w,a,out,"striped16");
  run<32,false>(x,w,a,out,"striped32");
  run<8,true>(x,w,a,out,"contiguous8");run<32,true>(x,w,a,out,"contiguous32");
  accurate<<<dim3(144,8),256>>>(x,w,a);
  check(cudaGetLastError());check(cudaDeviceSynchronize());save(a,144*8192,out+"/float64.bf16");
  cublasHandle_t handle;blas(cublasCreate(&handle));
  const float alpha=1.0f,beta=0.0f;
  for(int row=0;row<144;++row)blas(cublasGemmStridedBatchedEx(handle,CUBLAS_OP_T,CUBLAS_OP_N,
    1024,1,4096,&alpha,w,CUDA_R_16BF,4096,4096LL*1024,
    x+row*32768,CUDA_R_16BF,4096,4096,&beta,a+row*8192,CUDA_R_16BF,1024,1024,8,
    CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT));
  check(cudaDeviceSynchronize());save(a,144*8192,out+"/vendor_one_row.bf16");
  blas(cublasDestroy(handle));
  }
  for(void *p:{(void*)x,(void*)wa,(void*)sa,(void*)w,(void*)a,(void*)counts})check(cudaFree(p));
  std::printf("output_a_accumulation_probe rows=%d groups=8 input=4096 rank=1024 release=balanced offline_only=true\n",live);
}
