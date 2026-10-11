// Exact native routed-expert lowering. Matched mode substitutes the original
// down operands; propagated mode retains the native gate/up result.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include "expert_matched_kernels.cuh"

static void check(cudaError_t e) {
  if(e!=cudaSuccess){std::fprintf(stderr,"%s\n",cudaGetErrorString(e));std::exit(2);}
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
static void host_load(const std::string &path,unsigned char *destination,size_t bytes) {
  auto *file=std::fopen(path.c_str(),"rb");if(!file){std::perror(path.c_str());std::exit(3);}
  bool ok=std::fread(destination,1,bytes,file)==bytes&&std::fgetc(file)==EOF;
  std::fclose(file);if(!ok)std::exit(3);
}
int main(int argc,char **argv) {
  if(argc!=4)return 2;
  const std::string input(argv[1]),out(argv[2]),mode(argv[3]);
  const bool matched=mode=="matched";if(!matched&&mode!="propagated")return 2;
  std::vector<unsigned char> host_bank(expert_stride*6,0);
  for(int slot=0;slot<6;++slot)for(int stage=0;stage<3;++stage) {
    const std::string stem=input+"/slot"+std::to_string(slot)+"-w"+std::to_string(stage);
    auto *target=host_bank.data()+expert_stride*slot+expert_offsets[stage];
    host_load(stem+".fp4",target,4194304);host_load(stem+".scale",target+4194304,262144);
  }
  auto *bank=allocate<unsigned char>(host_bank.size());
  check(cudaMemcpy(bank,host_bank.data(),host_bank.size(),cudaMemcpyHostToDevice));
  auto *x=load<__nv_bfloat16>(input+"/input.bf16",4096);
  auto *weights=load<float>(input+"/scores.f32",6);
  auto *counts=allocate<int>(5),*indices=allocate<int>(6),*map=allocate<int>(6);
  int host_counts[5]={0,0,0,1,0},host_indices[6]={0,1,2,3,4,5};
  check(cudaMemcpy(counts,host_counts,sizeof(host_counts),cudaMemcpyHostToDevice));
  check(cudaMemcpy(indices,host_indices,sizeof(host_indices),cudaMemcpyHostToDevice));
  check(cudaMemcpy(map,host_indices,sizeof(host_indices),cudaMemcpyHostToDevice));
  auto *q0=allocate<unsigned char>(quantized_bytes[0]),*q1=allocate<unsigned char>(quantized_bytes[1]);
  auto *activation=allocate<__nv_bfloat16>(6*2048),*outputs=allocate<__nv_bfloat16>(6*4096);
  auto *error=allocate<int>(1);
  matched_expert_quantize_0<<<32,128>>>(counts,x,q0,error);
  matched_expert_gate_up<<<3072,128>>>(counts,q0,indices,map,bank,activation,weights);
  check(cudaGetLastError());check(cudaDeviceSynchronize());
  for(int slot=0;slot<6;++slot)save(activation+slot*2048,2048,out+"/slot"+std::to_string(slot)+"-activation.bf16");
  if(matched) {
    std::vector<unsigned char> host_activation(6*2048*2);
    for(int slot=0;slot<6;++slot)host_load(input+"/slot"+std::to_string(slot)+"-activation.bf16",host_activation.data()+slot*4096,4096);
    check(cudaMemcpy(activation,host_activation.data(),host_activation.size(),cudaMemcpyHostToDevice));
  }
  matched_expert_quantize_1<<<96,128>>>(counts,activation,q1,error);
  matched_expert_down<<<6144,128>>>(counts,q1,indices,weights,map,bank,outputs);
  check(cudaGetLastError());check(cudaDeviceSynchronize());
  for(int slot=0;slot<6;++slot)save(outputs+slot*4096,4096,out+"/slot"+std::to_string(slot)+"-output.bf16");
  int host_error=0;check(cudaMemcpy(&host_error,error,sizeof(int),cudaMemcpyDeviceToHost));
  if(host_error){std::fprintf(stderr,"nonfinite quantizer input %d\n",host_error);return 5;}
  for(void *p:{(void*)bank,(void*)x,(void*)weights,(void*)counts,(void*)indices,(void*)map,(void*)q0,(void*)q1,(void*)activation,(void*)outputs,(void*)error})check(cudaFree(p));
  std::puts("native_experts_complete route_mapping=original_slot_order release=balanced full_model=false");
}

