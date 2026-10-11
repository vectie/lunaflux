// Real rotary/cache/window lowering, literal original operands at each boundary.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include "window_matched_kernels.cuh"

static void check(cudaError_t e) {
  if(e!=cudaSuccess){std::fprintf(stderr,"%s\n",cudaGetErrorString(e));std::exit(2);}
}
template<class T> static T *allocate(size_t n) {
  T *p=nullptr;check(cudaMalloc(&p,n*sizeof(T)));check(cudaMemset(p,0,n*sizeof(T)));return p;
}
template<class T> static T *load(const std::string &path,size_t n) {
  auto *file=std::fopen(path.c_str(),"rb");if(!file){std::perror(path.c_str());std::exit(3);}
  std::vector<T> values(n);const bool read=std::fread(values.data(),sizeof(T),n,file)==n&&std::fgetc(file)==EOF;
  std::fclose(file);if(!read)std::exit(3);
  T *p=allocate<T>(n);check(cudaMemcpy(p,values.data(),n*sizeof(T),cudaMemcpyHostToDevice));return p;
}
template<class T> static void save(T *p,size_t n,const std::string &path) {
  std::vector<T> values(n);check(cudaMemcpy(values.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost));
  auto *file=std::fopen(path.c_str(),"wbx");if(!file){std::perror(path.c_str());std::exit(4);}
  const bool wrote=std::fwrite(values.data(),sizeof(T),n,file)==n;
  const bool closed=std::fclose(file)==0;if(!wrote||!closed)std::exit(4);
}
int main(int argc,char **argv) {
  if(argc!=5)return 2;
  const int rows=std::atoi(argv[3]);if(rows<128||rows>144)return 2;
  const std::string input(argv[1]),out(argv[2]),mode(argv[4]);
  const bool matched=mode=="matched";if(!matched&&mode!="propagated")return 2;
  auto *q=load<uint16_t>(input+"/q-input.bf16",rows*32768),*kv=load<uint16_t>(input+"/kv-input.bf16",rows*512);
  auto *sink=load<float>(input+"/sink.f32",64);
  auto *counts=allocate<int>(5),*positions=allocate<int>(rows),*reset=allocate<int>(1),
    *history=allocate<int>(2),*append=allocate<int>(3);
  int host[5]={0,0,1,rows,0},flag=1;std::vector<int> pos(rows);for(int i=0;i<rows;++i)pos[i]=i;
  check(cudaMemcpy(counts,host,sizeof(host),cudaMemcpyHostToDevice));
  check(cudaMemcpy(reset,&flag,sizeof(flag),cudaMemcpyHostToDevice));
  check(cudaMemcpy(positions,pos.data(),rows*sizeof(int),cudaMemcpyHostToDevice));
  const char *stages[5]={"q-rope","kv-rope","kv-sim","attention","inverse"};
  const int widths[5]={32768,512,512,32768,32768};uint16_t *buffers[5],*original[5]={};
  for(int i=0;i<5;++i){buffers[i]=allocate<uint16_t>(rows*widths[i]);
    if(matched)original[i]=load<uint16_t>(input+"/"+stages[i]+".bf16",rows*widths[i]);}
  auto *ring=allocate<__nv_bfloat16>(128*512);
  matched_window_rotary<<<dim3(rows,65),256>>>(counts,positions,q,kv,buffers[0],buffers[1]);
  check(cudaMemcpy(buffers[2],matched?original[1]:buffers[1],rows*512*2,cudaMemcpyDeviceToDevice));
  matched_window_simulate<<<dim3(rows,7),64>>>(counts,reinterpret_cast<__nv_bfloat16*>(buffers[2]));
  matched_window_reserve<<<1,1>>>(counts,reset,positions,history,append);
  matched_window_attention<<<dim3(rows,64),128>>>(append,
    reinterpret_cast<__nv_bfloat16*>(matched?original[0]:buffers[0]),
    reinterpret_cast<__nv_bfloat16*>(matched?original[2]:buffers[2]),ring,sink,
    reinterpret_cast<__nv_bfloat16*>(buffers[3]));
  matched_window_inverse<<<dim3(rows,64),256>>>(counts,positions,matched?original[3]:buffers[3],buffers[4]);
  matched_window_copy<<<128,128>>>(append,matched?original[2]:buffers[2],reinterpret_cast<uint16_t*>(ring));
  matched_window_publish<<<1,1>>>(append,history);
  check(cudaGetLastError());check(cudaDeviceSynchronize());
  int retained[2];check(cudaMemcpy(retained,history,sizeof(retained),cudaMemcpyDeviceToHost));
  if(retained[0]!=rows||retained[1]!=0)return 5;
  for(int i=0;i<5;++i){save(buffers[i],rows*widths[i],out+"/"+stages[i]+".bf16");
    check(cudaFree(buffers[i]));if(original[i])check(cudaFree(original[i]));}
  for(void *p:{(void*)q,(void*)kv,(void*)sink,(void*)counts,(void*)positions,(void*)reset,
      (void*)history,(void*)append,(void*)ring})check(cudaFree(p));
  std::printf("native_window_complete rows=%d mode=%s retained=%d release=balanced full_model=false\n",rows,mode.c_str(),retained[0]);
}
