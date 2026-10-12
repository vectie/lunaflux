// Run production routing lowering on the same literal original FFN corpus.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include "router_matched_kernels.cuh"
static void check(cudaError_t e){if(e!=cudaSuccess){std::fprintf(stderr,"%s\n",cudaGetErrorString(e));std::exit(2);}}
template<class T> static T *allocate(size_t n){T *p=nullptr;check(cudaMalloc(&p,n*sizeof(T)));check(cudaMemset(p,0,n*sizeof(T)));return p;}
template<class T> static T *load(const std::string &path,size_t n){
  auto *file=std::fopen(path.c_str(),"rb");if(!file){std::perror(path.c_str());std::exit(3);}
  std::vector<T> v(n);const bool ok=std::fread(v.data(),sizeof(T),n,file)==n&&std::fgetc(file)==EOF;
  std::fclose(file);if(!ok)std::exit(3);auto *p=allocate<T>(n);
  check(cudaMemcpy(p,v.data(),n*sizeof(T),cudaMemcpyHostToDevice));return p;
}
template<class T> static void save(T *p,size_t n,const std::string &path){
  std::vector<T> v(n);check(cudaMemcpy(v.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost));
  auto *file=std::fopen(path.c_str(),"wbx");if(!file){std::perror(path.c_str());std::exit(4);}
  const bool ok=std::fwrite(v.data(),sizeof(T),n,file)==n;
  const bool closed=std::fclose(file)==0;if(!ok||!closed)std::exit(4);
}
int main(int argc,char **argv){
  if(argc!=5&&argc!=6)return 2;
  const bool striped=argc==6&&std::string(argv[5])=="--striped32";if(argc==6&&!striped)return 2;
  const std::string input(argv[1]),ids_path(argv[2]),out(argv[3]);
  const int layer=std::atoi(argv[4]);if(layer!=0&&layer!=3)return 2;
  auto *x=load<__nv_bfloat16>(input+"/../input.bf16",144*4096);
  auto *w=load<__nv_bfloat16>(input+"/weight.bf16",256*4096);
  auto *ids=load<int>(ids_path,144),*counts=allocate<int>(5),*indices=allocate<int>(144*6);
  const int live[5]={0,0,0,144,0};check(cudaMemcpy(counts,live,sizeof(live),cudaMemcpyHostToDevice));
  auto *logits=allocate<float>(144*256),*scores=allocate<float>(144*256),*choice=allocate<float>(144*256),*weights=allocate<float>(144*6);
  int *table=nullptr;float *bias=nullptr;
  if(layer==0){
    table=load<int>(input+"/table.i32",129280*6);
    matched_hash_project<<<striped?36864:144,striped?32:256>>>(counts,x,w,logits);
    matched_hash_score<<<144,256>>>(counts,logits,scores);
    matched_hash_lookup<<<144,256>>>(counts,ids,table,indices);
    matched_hash_gather<<<144,1>>>(counts,scores,indices,weights);
  }else{
    bias=load<float>(input+"/bias.f32",256);
    matched_learned_project<<<striped?36864:144,striped?32:256>>>(counts,x,w,logits);
    matched_learned_score<<<144,256>>>(counts,logits,bias,scores,choice);
    matched_learned_choose<<<144,1>>>(counts,scores,choice,indices,weights);
  }
  check(cudaGetLastError());check(cudaDeviceSynchronize());
  save(logits,144*256,out+"/logits.f32");save(scores,144*256,out+"/scores.f32");
  save(indices,144*6,out+"/indices.i32");save(weights,144*6,out+"/weights.f32");
  for(void *p:{(void*)x,(void*)w,(void*)ids,(void*)counts,(void*)indices,(void*)logits,(void*)scores,(void*)choice,(void*)weights,(void*)table,(void*)bias})if(p)check(cudaFree(p));
  std::puts("native_router_complete rows=144 release=balanced full_model=false");
}
