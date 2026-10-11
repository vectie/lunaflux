// Offline exact-source probability intervention on captured native Q/K/V.
#include "main.cuh"
#include <fstream>
#include <string>
#include <vector>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <cstdlib>

static void check(cudaError_t error) {
  if(error!=cudaSuccess){std::fprintf(stderr,"%s\n",cudaGetErrorString(error));std::exit(2);}
}
static std::vector<char> read(const std::string &path,size_t bytes) {
  std::ifstream file(path,std::ios::binary);
  std::vector<char> data((std::istreambuf_iterator<char>(file)),{});
  if(!file||data.size()!=bytes){std::fprintf(stderr,"literal extent differs: %s\n",path.c_str());std::exit(3);}
  return data;
}
static void save(const std::string &path,const void *data,size_t bytes) {
  auto *file=std::fopen(path.c_str(),"wbx");
  if(!file||std::fwrite(data,1,bytes,file)!=bytes||std::fclose(file)!=0)std::exit(4);
}
int main(int argc,char **argv) {
  if(argc!=4&&argc!=5)return 2;
  const bool write=argc==4;
  if(!write&&std::string(argv[4])!="--verify-only")return 2;
  constexpr size_t cells=3*56*128,bytes=cells*2;
  __nv_bfloat16 *input=nullptr,*output=nullptr;
  check(cudaMallocManaged(&input,bytes*3));check(cudaMallocManaged(&output,bytes));
  for(int layer=0;layer<2;++layer) {
    const std::string prefix="refiner-"+std::to_string(layer)+"-";
    const auto operands=read(std::string(argv[1])+"/"+prefix+"qk-norm.bf16",bytes*3);
    const auto native=read(std::string(argv[1])+"/"+prefix+"attention.bf16",bytes);
    const auto original=read(std::string(argv[2])+"/"+prefix+"attention.bf16",bytes);
    std::memcpy(input,operands.data(),operands.size());
    for(int mode=0;mode<2;++mode) {
      if(mode)refiner_bf16_attention<<<(cells+255)/256,256>>>(input,input+cells,input+cells*2,output);
      else refiner_f32_attention<<<(cells+255)/256,256>>>(input,input+cells,input+cells*2,output);
      check(cudaGetLastError());check(cudaDeviceSynchronize());
      size_t exact=0;
      for(size_t i=0;i<cells;++i) {
        if(!std::isfinite(__bfloat162float(output[i])))std::exit(5);
        exact+=std::memcmp(output+i,original.data()+i*2,2)==0;
      }
      if(!mode&&std::memcmp(output,native.data(),bytes)!=0){std::fprintf(stderr,"unchanged F32 route not reproduced\n");std::exit(6);}
      if(write)save(std::string(argv[3])+"/"+prefix+(mode?"bf16":"f32")+".bf16",output,bytes);
      std::printf("layer=%d law=%s original_exact=%zu elements=%zu native_f32_preserved=true\n",layer,mode?"global_unnormalized_bf16":"global_f32",exact,cells);
    }
  }
  check(cudaFree(output));check(cudaFree(input));
  std::puts("probability_intervention_complete full_conditioning_and_media_unproven=true release=balanced");
}
