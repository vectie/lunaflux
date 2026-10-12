// Offline matched controls and internal arithmetic from actual renderer.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include "hyper_arithmetic_kernels.cuh"
static void check(cudaError_t e) {
  if(e!=cudaSuccess){std::fprintf(stderr,"%s\n",cudaGetErrorString(e));std::exit(2);}
}
template<class T> static T *load(const std::string &path,size_t n) {
  auto *file=std::fopen(path.c_str(),"rb");if(!file){std::perror(path.c_str());std::exit(3);}
  std::vector<T> values(n);const bool read=std::fread(values.data(),sizeof(T),n,file)==n&&std::fgetc(file)==EOF;
  std::fclose(file);if(!read){std::fprintf(stderr,"incomplete %s\n",path.c_str());std::exit(3);}
  T *p=nullptr;check(cudaMalloc(&p,n*sizeof(T)));check(cudaMemcpy(p,values.data(),n*sizeof(T),cudaMemcpyHostToDevice));return p;
}
template<class T> static T *allocate(size_t n) {
  T *p=nullptr;check(cudaMalloc(&p,n*sizeof(T)));check(cudaMemset(p,0,n*sizeof(T)));return p;
}
template<class T> static void save(T *p,size_t n,const std::string &path) {
  std::vector<T> values(n);check(cudaMemcpy(values.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost));
  auto *file=std::fopen(path.c_str(),"wbx");if(!file){std::perror(path.c_str());std::exit(4);}
  const bool wrote=std::fwrite(values.data(),sizeof(T),n,file)==n;const bool closed=std::fclose(file)==0;
  if(!wrote||!closed)std::exit(4);
}
int main(int argc,char **argv) {
  if(argc!=3)return 2;
  const std::string input(argv[1]),out(argv[2]);
  auto *x=load<__nv_bfloat16>(input+"/streams.bf16",144*16384);
  auto *fn=load<float>(input+"/function.f32",24*16384),*base=load<float>(input+"/base.f32",24),*scale=load<float>(input+"/scale.f32",3);
  auto *literal=load<float>(input+"/mixes.f32",144*24),*literal_inverse=load<float>(input+"/rsqrt.f32",144);
  auto *norm=load<__nv_bfloat16>(input+"/norm.bf16",4096);
  auto *mixes=allocate<float>(144*24),*inverse=allocate<float>(144),*post=allocate<float>(144*4);
  auto *initial=allocate<float>(144*16),*comb=allocate<float>(144*16);
  auto *collapsed=allocate<__nv_bfloat16>(144*4096);
  auto *normalized=allocate<__nv_bfloat16>(144*4096);
  auto *counts=allocate<int>(5);int host[5]={0,0,0,144,0};
  check(cudaMemcpy(counts,host,sizeof(host),cudaMemcpyHostToDevice));
#define RUN_ENVELOPE(symbol,sinkhorn_symbol,norm_symbol,threads,label) do { \
  symbol<<<144,1>>>(counts,x,fn,base,scale,literal,literal_inverse,mixes,inverse,post,initial,collapsed); \
  sinkhorn_symbol<<<144,1>>>(counts,initial,comb); \
  norm_symbol<<<144,threads>>>(counts,collapsed,norm,normalized); \
  check(cudaGetLastError());check(cudaDeviceSynchronize()); \
  save(mixes,144*24,out+"/" label "-mixes.f32");save(inverse,144,out+"/" label "-rsqrt.f32"); \
  save(post,144*4,out+"/" label "-post.f32");save(comb,144*16,out+"/" label "-combination.f32"); \
  save(collapsed,144*4096,out+"/" label "-collapsed.bf16");save(normalized,144*4096,out+"/" label "-normalized.bf16"); \
} while(false)
#define RUN_NORM(symbol,norm_symbol,threads,label) RUN_ENVELOPE(symbol,hc_probe_sinkhorn,norm_symbol,threads,label)
#define RUN(symbol,label) RUN_NORM(symbol,hc_probe_norm,256,label)
  RUN(hc_probe_function,"production");
  RUN(hc_refined_function,"refined");
  RUN(hc_nonlinear_function,"nonlinear");
  RUN_NORM(hc_vector_rms_function,hc_vector_rms_norm,512,"vector_rms");
  RUN_NORM(hc_hierarchical_function,hc_hierarchical_norm,512,"hierarchical");
  RUN_ENVELOPE(hc_butterfly_function,hc_butterfly_sinkhorn,hc_butterfly_norm,512,"butterfly");
  RUN(hc_refined_direct_function,"refined_direct");
  RUN(hc_refined_affine_fma_function,"refined_affine_fma");
  RUN(hc_refined_direct_affine_fma_function,"refined_direct_affine_fma");
  RUN(hc_refined_rsqrt_function,"refined_rsqrt");
  RUN(hc_literal_mix_function,"literal_mix");
  RUN(hc_literal_mix_direct_function,"literal_mix_direct");
  RUN(hc_literal_mix_affine_fma_function,"literal_mix_affine_fma");
  RUN(hc_literal_mix_direct_affine_fma_function,"literal_mix_direct_affine_fma");
  RUN(hc_literal_inverse_function,"literal_inverse");
  RUN(hc_striped32_function,"striped32");
  RUN(hc_striped32_fma_function,"striped32_fma");
  RUN(hc_striped32_literal_inverse_function,"striped32_literal_inverse");
  RUN(hc_striped32_fma_literal_inverse_function,"striped32_fma_literal_inverse");
  RUN(hc_compensated_square_function,"compensated_square");
  RUN(hc_square256_function,"square256");
  RUN(hc_striped32_fma_compensated_square_function,"striped32_fma_compensated_square");
  RUN(hc_striped32_fma_square256_function,"striped32_fma_square256");
  for(void *p:{(void*)x,(void*)fn,(void*)base,(void*)scale,(void*)literal,(void*)literal_inverse,
    (void*)mixes,(void*)inverse,(void*)post,(void*)initial,(void*)comb,(void*)collapsed,(void*)norm,(void*)normalized,(void*)counts})check(cudaFree(p));
  std::puts("hyper_arithmetic_probe rows=144 original_internal_operands=true production_control=true offline_only=true release=balanced");
}
