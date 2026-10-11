// Diagnostic: the unchanged 2,816-row generated translation unit, with only
// its first attention CTA submitted. This is not whole-domain sanitizer proof.
#include "dit_n1.cu"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <vector>

static void check(cudaError_t error) {
  if(error!=cudaSuccess){std::fprintf(stderr,"%s\n",cudaGetErrorString(error));std::exit(2);}
}
static std::vector<char> read(const char *path,size_t bytes) {
  std::ifstream file(path,std::ios::binary);
  std::vector<char> value((std::istreambuf_iterator<char>(file)),{});
  if(!file||value.size()!=bytes){std::fprintf(stderr,"literal capture extent differs\n");std::exit(3);}
  return value;
}
int main(int argc,char **argv) {
  if(argc!=3)return 2;
  constexpr size_t cells=2816ULL*56*128,cta_cells=256;
  const auto operands=read(argv[1],cells*3*2);
  const auto expected=read(argv[2],cells*2);
  __nv_bfloat16 *qkv=nullptr,*output=nullptr;
  check(cudaMallocManaged(&qkv,operands.size()));
  check(cudaMallocManaged(&output,cta_cells*2));
  std::memcpy(qkv,operands.data(),operands.size());
  cudaEvent_t begin,end;check(cudaEventCreate(&begin));check(cudaEventCreate(&end));
  float total=0.0f;
  for(int replay=0;replay<10;++replay) {
    check(cudaEventRecord(begin));
    h3_block_n1_attn_attention<<<1,256>>>(qkv,qkv+cells,qkv+2*cells,output);
    check(cudaGetLastError());check(cudaEventRecord(end));check(cudaEventSynchronize(end));
    float elapsed=0.0f;check(cudaEventElapsedTime(&elapsed,begin,end));total+=elapsed;
    if(std::memcmp(output,expected.data(),cta_cells*2)!=0) {
      std::fprintf(stderr,"executed candidate attention CTA not reproduced\n");return 4;
    }
  }
  check(cudaEventDestroy(end));check(cudaEventDestroy(begin));
  check(cudaFree(output));check(cudaFree(qkv));
  std::printf("scope=first_attention_cta_only exact_bf16=256 replays=10 mean_millis=%g release=balanced full_domain_sanitizer=false production_selection_changed=false\n",total/10.0f);
}
