// The exact exported production envelope on literal original component inputs.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include "hyper_matched_kernels.cuh"

static void check(cudaError_t e) {
  if(e!=cudaSuccess){std::fprintf(stderr,"%s\n",cudaGetErrorString(e));std::exit(2);}
}
template<class T> static T *load(const std::string &path,size_t n) {
  auto *file=std::fopen(path.c_str(),"rb");
  if(!file){std::perror(path.c_str());std::exit(3);}
  std::vector<T> values(n);
  const bool read=std::fread(values.data(),sizeof(T),n,file)==n&&std::fgetc(file)==EOF;
  std::fclose(file);
  if(!read){std::fprintf(stderr,"incomplete %s\n",path.c_str());std::exit(3);}
  T *p=nullptr;check(cudaMalloc(&p,n*sizeof(T)));
  check(cudaMemcpy(p,values.data(),n*sizeof(T),cudaMemcpyHostToDevice));return p;
}
template<class T> static T *allocate(size_t n) {
  T *p=nullptr;check(cudaMalloc(&p,n*sizeof(T)));check(cudaMemset(p,0,n*sizeof(T)));return p;
}
template<class T> static void save(T *p,size_t n,const std::string &path) {
  std::vector<T> values(n);check(cudaMemcpy(values.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost));
  auto *file=std::fopen(path.c_str(),"wbx");
  if(!file){std::perror(path.c_str());std::exit(4);}
  const bool wrote=std::fwrite(values.data(),sizeof(T),n,file)==n;
  const bool closed=std::fclose(file)==0;if(!wrote||!closed)std::exit(4);
}
int main(int argc,char **argv) {
  if(argc!=4&&argc!=5)return 2;
  const int rows=std::atoi(argv[3]);if(rows<1||rows>144)return 2;
  const std::string input(argv[1]),out(argv[2]);
  auto *streams=load<__nv_bfloat16>(input+"/streams.bf16",rows*16384);
  auto *fn=load<float>(input+"/function.f32",24*16384);
  auto *base=load<float>(input+"/base.f32",24),*scale=load<float>(input+"/scale.f32",3);
  auto *norm=load<__nv_bfloat16>(input+"/norm.bf16",4096);
  auto *counts=allocate<int>(5);int host[5]={0,0,0,rows,0};
  check(cudaMemcpy(counts,host,sizeof(host),cudaMemcpyHostToDevice));
  auto *collapsed=allocate<__nv_bfloat16>(rows*4096),*normalized=allocate<__nv_bfloat16>(rows*4096);
  auto *post=allocate<float>(rows*4),*initial=allocate<float>(rows*16),*comb=allocate<float>(rows*16);
  matched_hyper_function<<<rows,1>>>(counts,streams,fn,base,scale,post,initial,collapsed);
  matched_hyper_sinkhorn<<<rows,1>>>(counts,initial,comb);
  matched_hyper_norm<<<rows,256>>>(counts,collapsed,norm,normalized);
  check(cudaGetLastError());check(cudaDeviceSynchronize());
  save(collapsed,rows*4096,out+"/collapsed.bf16");save(normalized,rows*4096,out+"/normalized.bf16");
  save(post,rows*4,out+"/post.f32");save(comb,rows*16,out+"/combination.f32");
  if(argc==5) {
    const std::string prior(argv[4]);
    auto *residual=load<__nv_bfloat16>(prior+"/streams.bf16",rows*16384);
    auto *branch=load<__nv_bfloat16>(prior+"/branch.bf16",rows*4096);
    auto *old_post=load<float>(prior+"/post.f32",rows*4);
    auto *old_comb=load<float>(prior+"/combination.f32",rows*16);
    auto *published=allocate<__nv_bfloat16>(rows*16384);
    matched_hyper_post<<<rows,256>>>(counts,branch,residual,old_post,old_comb,published);
    check(cudaGetLastError());check(cudaDeviceSynchronize());
    save(published,rows*16384,out+"/attention-post.bf16");
    for(void *p:{(void*)residual,(void*)branch,(void*)old_post,(void*)old_comb,(void*)published})check(cudaFree(p));
  }
  for(void *p:{(void*)streams,(void*)fn,(void*)base,(void*)scale,(void*)norm,(void*)counts,
    (void*)collapsed,(void*)normalized,(void*)post,(void*)initial,(void*)comb})check(cudaFree(p));
  std::printf("native_hyper_complete rows=%d literal_inputs=true release=balanced full_model=false\n",rows);
}
