// Numerical intervention only: identical production operands, unchanged source
// versus an offline intrinsic-exp experiment. Neither alters serving selection.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#ifdef LUNA_EXPERIMENT_FAST_EXP
#define expf __expf
#endif
#include "window_matched_kernels.cuh"
static void check(cudaError_t e){if(e!=cudaSuccess){std::fprintf(stderr,"%s\n",cudaGetErrorString(e));std::exit(2);}}
template<class T> static T *allocate(size_t n){T *p=nullptr;check(cudaMalloc(&p,n*sizeof(T)));check(cudaMemset(p,0,n*sizeof(T)));return p;}
template<class T> static T *load(const std::string &path,size_t n){
  auto *f=std::fopen(path.c_str(),"rb");if(!f)std::exit(3);
  std::vector<T> x(n);const bool ok=std::fread(x.data(),sizeof(T),n,f)==n&&std::fgetc(f)==EOF;
  std::fclose(f);if(!ok)std::exit(3);auto *p=allocate<T>(n);check(cudaMemcpy(p,x.data(),n*sizeof(T),cudaMemcpyHostToDevice));return p;
}
int main(int argc,char **argv){
  if(argc!=3)return 2;
  const std::string input(argv[1]),output(argv[2]);
  auto *q=load<__nv_bfloat16>(input+"/q-rope.bf16",144*32768),*kv=load<__nv_bfloat16>(input+"/kv-sim.bf16",144*512);
  auto *sink=load<float>(input+"/sink.f32",64);
  auto *ring=allocate<__nv_bfloat16>(128*512),*out=allocate<__nv_bfloat16>(144*32768);
  auto *append=allocate<int>(3);const int host[3]={0,144,0};check(cudaMemcpy(append,host,sizeof(host),cudaMemcpyHostToDevice));
  matched_window_attention<<<dim3(144,64),128>>>(append,q,kv,ring,sink,out);
  check(cudaGetLastError());check(cudaDeviceSynchronize());
  std::vector<uint16_t> values(144*32768);check(cudaMemcpy(values.data(),out,values.size()*2,cudaMemcpyDeviceToHost));
  auto *f=std::fopen((output+"/attention.bf16").c_str(),"wbx");if(!f)return 4;
  const bool ok=std::fwrite(values.data(),2,values.size(),f)==values.size();const bool closed=std::fclose(f)==0;if(!ok||!closed)return 4;
  for(void *p:{(void*)q,(void*)kv,(void*)sink,(void*)ring,(void*)out,(void*)append})check(cudaFree(p));
  std::puts("native_window_literal_operands completed=true release=balanced");
}
