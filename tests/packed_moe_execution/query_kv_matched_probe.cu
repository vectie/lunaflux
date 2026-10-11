// Exact native query/KV lowering. Matched mode substitutes each literal original
// operand separately; propagated mode is a complete native prefix on that input.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include "query_matched_kernels.cuh"

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
  auto *x=load<__nv_bfloat16>(input+"/input.bf16",rows*4096);
  const int in[3]={4096,1024,4096},width[3]={1024,32768,512};
  const char *matrix[3]={"wq_a","wq_b","wkv"};
  unsigned char *weights[3],*scales[3];
  for(int i=0;i<3;++i){
    weights[i]=load<unsigned char>(input+"/"+matrix[i]+".fp8",in[i]*width[i]);
    scales[i]=load<unsigned char>(input+"/"+matrix[i]+".scale",in[i]/128*(width[i]/128));
  }
  auto *qw=load<__nv_bfloat16>(input+"/q_norm.bf16",1024),*kw=load<__nv_bfloat16>(input+"/kv_norm.bf16",512);
  auto *counts=allocate<int>(5);int host[5]={0,0,0,rows,0};
  check(cudaMemcpy(counts,host,sizeof(host),cudaMemcpyHostToDevice));
  const int widths[6]={1024,1024,32768,32768,512,512};
  const char *stages[6]={"q-a","q-norm","q-b","q-head","kv","kv-norm"};
  __nv_bfloat16 *buffers[6],*original[6]={};
  for(int i=0;i<6;++i){
    buffers[i]=allocate<__nv_bfloat16>(rows*widths[i]);
    if(matched)original[i]=load<__nv_bfloat16>(input+"/"+stages[i]+".bf16",rows*widths[i]);
  }
  matched_query_q_a<<<dim3(rows,4),256>>>(counts,x,weights[0],scales[0],buffers[0]);
  matched_query_q_norm<<<rows,256>>>(counts,matched?original[0]:buffers[0],qw,buffers[1]);
  matched_query_q_b<<<dim3(rows,128),256>>>(counts,matched?original[1]:buffers[1],weights[1],scales[1],buffers[2]);
  matched_query_q_head_norm<<<dim3(rows,64),256>>>(counts,matched?original[2]:buffers[2],buffers[3]);
  matched_query_kv<<<dim3(rows,2),256>>>(counts,x,weights[2],scales[2],buffers[4]);
  matched_query_kv_norm<<<rows,256>>>(counts,matched?original[4]:buffers[4],kw,buffers[5]);
  check(cudaGetLastError());check(cudaDeviceSynchronize());
  for(int i=0;i<6;++i){
    save(buffers[i],rows*widths[i],out+"/"+stages[i]+".bf16");check(cudaFree(buffers[i]));
    if(original[i])check(cudaFree(original[i]));
  }
  for(int i=0;i<3;++i){check(cudaFree(weights[i]));check(cudaFree(scales[i]));}
  for(void *p:{(void*)x,(void*)qw,(void*)kw,(void*)counts})check(cudaFree(p));
  std::printf("native_query_kv_complete rows=%d mode=%s release=balanced full_model=false\n",rows,mode.c_str());
}
