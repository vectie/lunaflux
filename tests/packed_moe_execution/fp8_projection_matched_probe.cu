// Offline matched output-B projection; consumes actual AOT launch metadata.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include "projection_matched_kernels.cuh"

static void check(cudaError_t e) {
  if(e!=cudaSuccess){std::fprintf(stderr,"%s\n",cudaGetErrorString(e));std::exit(2);}
}
template<class T> static T *load(const std::string &path,size_t n,size_t corpus=0) {
  auto *file=std::fopen(path.c_str(),"rb");if(!file){std::perror(path.c_str());std::exit(3);}
  std::vector<T> values(n);
  if(corpus==0)corpus=n;
  if(std::fseek(file,0,SEEK_END)!=0||std::ftell(file)!=(long)(corpus*sizeof(T))||std::fseek(file,0,SEEK_SET)!=0)std::exit(3);
  const bool read=std::fread(values.data(),sizeof(T),n,file)==n;
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
  if(argc!=6)return 2;
  const int rows=std::atoi(argv[3]),corpus=std::atoi(argv[5]);
  if(rows<1||rows>144||corpus<rows||corpus>144)return 2;
  const std::string root(argv[1]),out(argv[2]);
  auto *x=load<__nv_bfloat16>(root+"/output-a.bf16",rows*8192,corpus*8192);
  auto *w=load<unsigned char>(root+"/wo_b.fp8",8192*4096);
  auto *ws=load<unsigned char>(root+"/wo_b.scale",64*32);
  auto *output=allocate<__nv_bfloat16>(rows*4096);
  auto *counts=allocate<int>(5);int values[5]={0,0,0,rows,0};
  check(cudaMemcpy(counts,values,sizeof(values),cudaMemcpyHostToDevice));
  matched_projection<<<dim3(rows,LUNA_PROJECTION_Y),LUNA_PROJECTION_THREADS>>>(counts,x,w,ws,output);
  check(cudaGetLastError());check(cudaDeviceSynchronize());
  save(output,rows*4096,out+"/output-b.bf16");
  for(void *p:{(void*)x,(void*)w,(void*)ws,(void*)output,(void*)counts})check(cudaFree(p));
  std::printf("matched_output_b rows=%d corpus=%d balanced=true complete_chain=false\n",rows,corpus);
}
