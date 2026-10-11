// Exact native query/KV lowering. Matched mode substitutes each literal original
// operand separately; propagated mode is a complete native prefix on that input.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include "output_matched_kernels.cuh"

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
int main(int argc,char **argv) {
  if(argc!=5)return 2;
  const int rows=std::atoi(argv[3]);if(rows<1||rows>144)return 2;
  const std::string input(argv[1]),out(argv[2]),mode(argv[4]);
  const bool matched=mode=="matched";if(!matched&&mode!="propagated")return 2;
  auto *x=load<__nv_bfloat16>(input+"/input.bf16",rows*32768);
  auto *wa=load<unsigned char>(input+"/wo_a.fp8",8192*4096);
  auto *sa=load<unsigned char>(input+"/wo_a.scale",64*32);
  auto *wb=load<unsigned char>(input+"/wo_b.fp8",4096*8192);
  auto *sb=load<unsigned char>(input+"/wo_b.scale",32*64);
  auto *a=allocate<__nv_bfloat16>(rows*8192),*b=allocate<__nv_bfloat16>(rows*4096);
  auto *original=matched?load<__nv_bfloat16>(input+"/output-a.bf16",rows*8192):nullptr;
  auto *counts=allocate<int>(5);int host[5]={0,0,0,rows,0};
  check(cudaMemcpy(counts,host,sizeof(host),cudaMemcpyHostToDevice));
  matched_output_grouped<<<dim3(rows,8),256>>>(counts,x,wa,sa,a);
  matched_output_output<<<dim3(rows,16),256>>>(counts,matched?original:a,wb,sb,b);
  check(cudaGetLastError());check(cudaDeviceSynchronize());
  save(a,rows*8192,out+"/output-a.bf16");save(b,rows*4096,out+"/output-b.bf16");
  for(void *p:{(void*)x,(void*)wa,(void*)sa,(void*)wb,(void*)sb,(void*)a,(void*)b,(void*)counts})check(cudaFree(p));
  if(original)check(cudaFree(original));
  std::printf("native_grouped_output_complete rows=%d mode=%s release=balanced full_model=false\\n",rows,mode.c_str());
}

