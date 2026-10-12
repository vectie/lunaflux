// OFFLINE causal boundaries: identical observed logits, scores and expert IDs.
// Vendor projection is an arithmetic control, never a production dependency.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
static void check(cudaError_t e){if(e!=cudaSuccess){std::fprintf(stderr,"%s\n",cudaGetErrorString(e));std::exit(2);}}
static void blas(cublasStatus_t e){if(e!=CUBLAS_STATUS_SUCCESS){std::fprintf(stderr,"blas=%d\n",int(e));std::exit(2);}}
template<class T> static T *allocate(size_t n){T *p=nullptr;check(cudaMalloc(&p,n*sizeof(T)));return p;}
template<class T> static T *load(const std::string &path,size_t n){
  auto *f=std::fopen(path.c_str(),"rb");if(!f){std::perror(path.c_str());std::exit(3);}
  std::vector<T> v(n);const bool ok=std::fread(v.data(),sizeof(T),n,f)==n&&std::fgetc(f)==EOF;
  std::fclose(f);if(!ok)std::exit(3);auto *p=allocate<T>(n);check(cudaMemcpy(p,v.data(),n*sizeof(T),cudaMemcpyHostToDevice));return p;
}
template<class T> static void save(T *p,size_t n,const std::string &path){
  std::vector<T> v(n);check(cudaMemcpy(v.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost));
  auto *f=std::fopen(path.c_str(),"wbx");if(!f){std::perror(path.c_str());std::exit(4);}
  const bool ok=std::fwrite(v.data(),sizeof(T),n,f)==n,closed=std::fclose(f)==0;if(!ok||!closed)std::exit(4);
}
__global__ void widen(const __nv_bfloat16 *x,float *y,int n){const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)y[i]=__bfloat162float(x[i]);}
// Observed vendor gemvx: one FMA chain per 32-way striped partition,
// then descending-half additions. Output packing is not part of that law.
__global__ void striped_fma(const float *x,const float *w,float *logits){
  const int row=blockIdx.x,expert=blockIdx.y,lane=threadIdx.x;
  __shared__ float sums[32];float sum=0;
  for(int k=lane;k<4096;k+=32)sum=__fmaf_rn(x[row*4096+k],w[expert*4096+k],sum);
  sums[lane]=sum;__syncthreads();
  for(int step=16;step>0;step/=2){if(lane<step)sums[lane]=__fadd_rn(sums[lane],sums[lane+step]);__syncthreads();}
  if(lane==0)logits[row*256+expert]=sums[0];
}
template<int Law> __global__ void score(const float *x,float *y){
  const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=144*256)return;
  const float a=x[i];float s;
  if(Law==0)s=a>0?__fadd_rn(a,log1pf(expf(-a))):log1pf(expf(a));
  else if(Law==1)s=a>20?a:log1pf(expf(a));
  else s=a>20?a:logf(__fadd_rn(1.0f,expf(a)));
  y[i]=sqrtf(s);
}
template<int Order,bool Reciprocal> __global__ void gather(const float *s,const int *ids,float *weights){
  const int r=blockIdx.x;if(threadIdx.x)return;float a[6];
  for(int i=0;i<6;++i)a[i]=s[r*256+ids[r*6+i]];
  float sum=0;
  if(Order==0){for(int i=0;i<6;++i)sum=__fadd_rn(sum,a[i]);}
  if(Order==1)sum=__fadd_rn(__fadd_rn(__fadd_rn(a[0],a[1]),__fadd_rn(a[2],a[3])),__fadd_rn(a[4],a[5]));
  if(Order==2)sum=__fadd_rn(__fadd_rn(__fadd_rn(a[0],a[3]),__fadd_rn(a[1],a[4])),__fadd_rn(a[2],a[5]));
  if(Order==3)sum=__fadd_rn(__fadd_rn(__fadd_rn(a[0],a[4]),a[2]),__fadd_rn(__fadd_rn(a[1],a[5]),a[3]));
  if(Order==4)sum=__fadd_rn(__fadd_rn(__fadd_rn(a[0],a[4]),__fadd_rn(a[1],a[5])),__fadd_rn(a[2],a[3]));
  for(int i=0;i<6;++i)weights[r*6+i]=__fmul_rn(Reciprocal?__fmul_rn(a[i],__fdiv_rn(1.0f,sum)):__fdiv_rn(a[i],sum),1.5f);
}
template<int O> static void weights(const float *s,const int *ids,float *w,const std::string &out,const std::string &scope){
  gather<O,false><<<144,1>>>(s,ids,w);check(cudaGetLastError());save(w,864,out+"/"+scope+"-sum"+std::to_string(O)+"-divide.f32");
  gather<O,true><<<144,1>>>(s,ids,w);check(cudaGetLastError());save(w,864,out+"/"+scope+"-sum"+std::to_string(O)+"-reciprocal.f32");
}
int main(int argc,char **argv){
  if(argc!=3&&argc!=4)return 2;
  const bool native_only=argc==4&&std::string(argv[3])=="--native-only";if(argc==4&&!native_only)return 2;
  const std::string root(argv[1]),out(argv[2]),original=root+"/original-layer-0";
  auto *x=load<__nv_bfloat16>(root+"/input.bf16",144*4096),*w=load<__nv_bfloat16>(original+"/weight.bf16",256*4096);
  auto *xf=allocate<float>(144*4096),*wf=allocate<float>(256*4096),*logits=allocate<float>(144*256);
  auto *observed_logits=load<float>(original+"/logits.f32",144*256),*observed_scores=load<float>(original+"/scores.f32",144*256);
  auto *ids=load<int>(original+"/indices.i32",864);
  auto *scores=allocate<float>(144*256),*route_weights=allocate<float>(864);
  widen<<<2304,256>>>(x,xf,144*4096);widen<<<4096,256>>>(w,wf,256*4096);check(cudaGetLastError());
  if(!native_only){
  cublasHandle_t handle;blas(cublasCreate(&handle));blas(cublasSetMathMode(handle,CUBLAS_PEDANTIC_MATH));const float alpha=1,beta=0;
  for(bool gemv:{true,false}){
    for(int r=0;r<144;++r){
      if(gemv)blas(cublasSgemv(handle,CUBLAS_OP_T,4096,256,&alpha,wf,4096,xf+r*4096,1,&beta,logits+r*256,1));
      else blas(cublasSgemm(handle,CUBLAS_OP_T,CUBLAS_OP_N,256,1,4096,&alpha,wf,4096,xf+r*4096,4096,&beta,logits+r*256,256));
    }
    save(logits,144*256,out+(gemv?"/vendor-gemv.f32":"/vendor-gemm.f32"));
  }
  blas(cublasDestroy(handle));
  }
  striped_fma<<<dim3(144,256),32>>>(xf,wf,logits);save(logits,144*256,out+"/native-striped32-fma.f32");
  score<1><<<144,256>>>(logits,scores);save(scores,144*256,out+"/native-chain-scores.f32");
  gather<3,false><<<144,1>>>(scores,ids,route_weights);save(route_weights,864,out+"/native-chain-weights.f32");
  score<0><<<144,256>>>(observed_logits,scores);save(scores,144*256,out+"/score-stable.f32");
  score<1><<<144,256>>>(observed_logits,scores);save(scores,144*256,out+"/score-direct-log1p.f32");
  score<2><<<144,256>>>(observed_logits,scores);save(scores,144*256,out+"/score-direct-log.f32");
  weights<0>(observed_scores,ids,route_weights,out,"original-score");weights<1>(observed_scores,ids,route_weights,out,"original-score");
  weights<2>(observed_scores,ids,route_weights,out,"original-score");weights<3>(observed_scores,ids,route_weights,out,"original-score");
  weights<4>(observed_scores,ids,route_weights,out,"original-score");
  check(cudaGetLastError());check(cudaDeviceSynchronize());
  for(void *p:{(void*)x,(void*)w,(void*)xf,(void*)wf,(void*)logits,(void*)observed_logits,(void*)observed_scores,(void*)ids,(void*)scores,(void*)route_weights})check(cudaFree(p));
  std::puts("router_arithmetic rows=144 observed_inputs=literal vendor=offline_control_only reference_changed=false release=balanced");
}
