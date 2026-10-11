// Production learned-pool lowering on literal original component operands.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include "pool_matched_kernels.cuh"
static void check(cudaError_t e) {
  if(e!=cudaSuccess){std::fprintf(stderr,"%s\n",cudaGetErrorString(e));std::exit(2);}
}
template<class T> static T *allocate(size_t n) {
  T *p=nullptr;check(cudaMalloc(&p,n*sizeof(T)));check(cudaMemset(p,0,n*sizeof(T)));return p;
}
template<class T> static T *load(const std::string &path,size_t n) {
  auto *file=std::fopen(path.c_str(),"rb");if(!file){std::perror(path.c_str());std::exit(3);}
  std::vector<T> values(n);const bool ok=std::fread(values.data(),sizeof(T),n,file)==n&&std::fgetc(file)==EOF;
  std::fclose(file);if(!ok)std::exit(3);
  T *p=allocate<T>(n);check(cudaMemcpy(p,values.data(),n*sizeof(T),cudaMemcpyHostToDevice));return p;
}
template<class T> static void replace(T *p,const std::string &path,size_t n) {
  auto *other=load<T>(path,n);check(cudaMemcpy(p,other,n*sizeof(T),cudaMemcpyDeviceToDevice));check(cudaFree(other));
}
template<class T> static void save(T *p,size_t n,const std::string &path) {
  std::vector<T> values(n);check(cudaMemcpy(values.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost));
  auto *file=std::fopen(path.c_str(),"wbx");if(!file){std::perror(path.c_str());std::exit(4);}
  bool ok=std::fwrite(values.data(),sizeof(T),n,file)==n;
  if(std::fclose(file)!=0||!ok)std::exit(4);
}
int main(int argc,char **argv) {
  if(argc!=5)return 2;
  const std::string input(argv[1]),out(argv[2]),mode(argv[4]);
  const bool matched=mode=="matched";if(!matched&&mode!="propagated")return 2;
  const int rows=144,completed=rows/pool_ratio;
  auto *x=load<__nv_bfloat16>(argv[3],rows*4096);
  auto *wkv=load<__nv_bfloat16>(input+"/wkv.bf16",pool_channels*4096);
  auto *wgate=load<__nv_bfloat16>(input+"/wgate.bf16",pool_channels*4096);
  auto *ape=load<float>(input+"/ape.f32",pool_ratio*pool_channels);
  auto *norm=load<__nv_bfloat16>(input+"/norm.bf16",512);
  auto *counts=allocate<int>(5),*positions=allocate<int>(rows),*reset=allocate<int>(1);
  auto *history=allocate<int>(2),*append=allocate<int>(6),*output_counts=allocate<int>(5),*output_positions=allocate<int>(completed+1);
  int host_counts[5]={0,0,1,rows,0},one=1;std::vector<int> host_positions(rows);
  for(int i=0;i<rows;++i)host_positions[i]=i;
  check(cudaMemcpy(counts,host_counts,sizeof(host_counts),cudaMemcpyHostToDevice));
  check(cudaMemcpy(reset,&one,sizeof(one),cudaMemcpyHostToDevice));
  check(cudaMemcpy(positions,host_positions.data(),rows*sizeof(int),cudaMemcpyHostToDevice));
  auto *kv=allocate<float>(rows*pool_channels),*scores=allocate<float>(rows*pool_channels);
  auto *kv_state=allocate<float>(pool_state_rows*pool_channels),*score_state=allocate<float>(pool_state_rows*pool_channels);
  auto *pooled=allocate<__nv_bfloat16>((completed+1)*512),*normalized=allocate<__nv_bfloat16>((completed+1)*512);
  matched_pool_reserve<<<1,1>>>(counts,positions,reset,history,append,output_counts,output_positions);
  matched_pool_project<<<dim3(rows,pool_channels/256),256>>>(append,x,wkv,wgate,kv,scores);
  check(cudaGetLastError());check(cudaDeviceSynchronize());
  save(kv,rows*pool_channels,out+"/wkv-projection.f32");save(scores,rows*pool_channels,out+"/wgate-projection.f32");
  if(matched){replace(kv,input+"/wkv-projection.f32",rows*pool_channels);replace(scores,input+"/wgate-projection.f32",rows*pool_channels);}
  matched_pool_pool<<<2,256>>>(append,kv,scores,ape,kv_state,score_state,pooled);
  check(cudaGetLastError());check(cudaDeviceSynchronize());save(pooled,completed*512,out+"/pooled.bf16");
  if(matched)replace(pooled,input+"/pooled.bf16",completed*512);
  matched_pool_norm<<<completed,256>>>(append,pooled,norm,normalized);
  matched_pool_publish<<<1,1>>>(append,history);
  check(cudaGetLastError());check(cudaDeviceSynchronize());save(normalized,completed*512,out+"/normalized.bf16");
  int host_history[2];check(cudaMemcpy(host_history,history,sizeof(host_history),cudaMemcpyDeviceToHost));
  if(host_history[0]!=rows||host_history[1])return 5;
  for(void *p:{(void*)x,(void*)wkv,(void*)wgate,(void*)ape,(void*)norm,(void*)counts,(void*)positions,
    (void*)reset,(void*)history,(void*)append,(void*)output_counts,(void*)output_positions,(void*)kv,
    (void*)scores,(void*)kv_state,(void*)score_state,(void*)pooled,(void*)normalized})check(cudaFree(p));
  std::printf("native_pool_complete ratio=%d rows=%d completed=%d release=balanced full_model=false\n",pool_ratio,rows,completed);
}
