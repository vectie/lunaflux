// The production attention renderer, on original literal post-rotary operands.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include "compressed_matched_kernels.cuh"
static void check(cudaError_t error) {
  if(error!=cudaSuccess){std::fprintf(stderr,"%s\n",cudaGetErrorString(error));std::exit(2);}
}
template<class T> static T *load(const std::string &path,size_t n) {
  auto *file=std::fopen(path.c_str(),"rb");if(!file){std::perror(path.c_str());std::exit(3);}
  std::vector<T> values(n);const bool read=std::fread(values.data(),sizeof(T),n,file)==n&&std::fgetc(file)==EOF;
  std::fclose(file);if(!read)std::exit(3);
  T *p=nullptr;check(cudaMalloc(&p,n*sizeof(T)));check(cudaMemcpy(p,values.data(),n*sizeof(T),cudaMemcpyHostToDevice));return p;
}
int main(int argc,char **argv) {
  if(argc!=3)return 2;
  const std::string input(argv[1]),out(argv[2]);
  auto *q=load<__nv_bfloat16>(input+"/q-rope.bf16",144*32768);
  auto *kv=load<__nv_bfloat16>(input+"/kv-sim.bf16",144*512);
  auto *compressed=load<__nv_bfloat16>(input+"/compressed.bf16",512);
  auto *sink=load<float>(input+"/sink.f32",64);
  int descriptor[3]={0,144,0};int *append=nullptr;
  check(cudaMalloc(&append,sizeof(descriptor)));check(cudaMemcpy(append,descriptor,sizeof(descriptor),cudaMemcpyHostToDevice));
  __nv_bfloat16 *result=nullptr;check(cudaMalloc(&result,144*32768*2));
  matched_compressed_attention<<<dim3(144,64),128>>>(append,nullptr,nullptr,nullptr,q,kv,nullptr,compressed,sink,result);
  check(cudaGetLastError());check(cudaDeviceSynchronize());
  std::vector<__nv_bfloat16> values(144*32768);check(cudaMemcpy(values.data(),result,values.size()*2,cudaMemcpyDeviceToHost));
  auto *file=std::fopen((out+"/attention.bf16").c_str(),"wbx");if(!file)return 4;
  const bool wrote=std::fwrite(values.data(),2,values.size(),file)==values.size();
  const bool closed=std::fclose(file)==0;if(!wrote||!closed)return 4;
  for(void *p:{(void*)q,(void*)kv,(void*)compressed,(void*)sink,(void*)append,(void*)result})check(cudaFree(p));
  std::puts("native_compressed_attention_complete rows=144 ratio=128 release=balanced full_model=false");
}
