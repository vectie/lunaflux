// Exact exported shared-expert lowering; substitute only literal original
// down operands in matched mode. No replacement model/kernel oracle.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include "shared_matched_kernels.cuh"
static void check(cudaError_t e) {
  if(e!=cudaSuccess){std::fprintf(stderr,"%s\n",cudaGetErrorString(e));std::exit(2);}
}
template<class T> static T *allocate(size_t n) {
  T *p=nullptr;check(cudaMalloc(&p,n*sizeof(T)));check(cudaMemset(p,0,n*sizeof(T)));return p;
}
static void host_load(const std::string &path,void *target,size_t bytes) {
  auto *file=std::fopen(path.c_str(),"rb");if(!file){std::perror(path.c_str());std::exit(3);}
  const bool ok=std::fread(target,1,bytes,file)==bytes&&std::fgetc(file)==EOF;
  std::fclose(file);if(!ok)std::exit(3);
}
template<class T> static T *load(const std::string &path,size_t n) {
  std::vector<T> host(n);host_load(path,host.data(),n*sizeof(T));auto *p=allocate<T>(n);
  check(cudaMemcpy(p,host.data(),n*sizeof(T),cudaMemcpyHostToDevice));return p;
}
template<class T> static void save(T *p,size_t n,const std::string &path) {
  std::vector<T> host(n);check(cudaMemcpy(host.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost));
  auto *file=std::fopen(path.c_str(),"wbx");if(!file){std::perror(path.c_str());std::exit(4);}
  const bool ok=std::fwrite(host.data(),sizeof(T),n,file)==n;const bool closed=std::fclose(file)==0;
  if(!ok||!closed)std::exit(4);
}
int main(int argc,char **argv) {
  if(argc!=4)return 2;const std::string input(argv[1]),out(argv[2]),mode(argv[3]);
  if(mode!="matched"&&mode!="propagated")return 2;
  std::vector<unsigned char> host_bank(shared_stride,0);
  for(int stage=0;stage<3;++stage) {
    const std::string name=stage==0?"w1":stage==1?"w3":"w2";
    auto *target=host_bank.data()+shared_offsets[stage];
    host_load(input+"/"+name+".fp8",target,8388608);
    host_load(input+"/"+name+".scale",target+8388608,512);
  }
  auto *bank=allocate<unsigned char>(shared_stride);
  check(cudaMemcpy(bank,host_bank.data(),host_bank.size(),cudaMemcpyHostToDevice));
  auto *x=load<__nv_bfloat16>(input+"/input.bf16",4096);
  auto *counts=allocate<int>(5),*indices=allocate<int>(1),*map=allocate<int>(1),*error=allocate<int>(1);
  const int host_counts[5]={0,0,0,1,0};check(cudaMemcpy(counts,host_counts,sizeof(host_counts),cudaMemcpyHostToDevice));
  auto *weights=allocate<float>(1);const float one=1;check(cudaMemcpy(weights,&one,4,cudaMemcpyHostToDevice));
  auto *q0=allocate<unsigned char>(shared_quantized_bytes[0]),*q1=allocate<unsigned char>(shared_quantized_bytes[1]);
  auto *activation=allocate<__nv_bfloat16>(2048),*output=allocate<__nv_bfloat16>(4096);
  matched_shared_quantize_0<<<32,128>>>(counts,x,q0,error);
  matched_shared_gate_up<<<512,128>>>(counts,q0,indices,map,bank,activation,weights);
  check(cudaGetLastError());check(cudaDeviceSynchronize());
  save(activation,2048,out+"/activation.bf16");
  if(mode=="matched") {
    std::vector<__nv_bfloat16> host(2048);host_load(input+"/activation.bf16",host.data(),4096);
    check(cudaMemcpy(activation,host.data(),4096,cudaMemcpyHostToDevice));
  }
  matched_shared_quantize_1<<<16,128>>>(counts,activation,q1,error);
  matched_shared_down<<<1024,128>>>(counts,q1,indices,weights,map,bank,output);
  check(cudaGetLastError());check(cudaDeviceSynchronize());save(output,4096,out+"/output.bf16");
  int host_error=0;check(cudaMemcpy(&host_error,error,4,cudaMemcpyDeviceToHost));
  for(void *p:{(void*)bank,(void*)x,(void*)counts,(void*)indices,(void*)map,(void*)error,(void*)weights,(void*)q0,(void*)q1,(void*)activation,(void*)output})check(cudaFree(p));
  if(host_error)return 5;std::puts("shared_expert_replay_complete original_operands=true release=balanced");
}
