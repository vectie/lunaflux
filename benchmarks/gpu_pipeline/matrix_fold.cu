// Isolated production-lowering comparison, including selected-row vocabulary.
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#define CK(x) do{auto e=(x);if(e){std::fprintf(stderr,"CUDA %d line %d\n",int(e),__LINE__);std::exit(2);}}while(0)
struct Buffer {
  void* p; size_t n;
  Buffer(size_t bytes):n(bytes){CK(cudaMalloc(&p,n));CK(cudaMemset(p,0,n));}
  ~Buffer(){CK(cudaFree(p));}
  std::vector<__nv_bfloat16> fill(int salt){
    std::vector<__nv_bfloat16> h(n/2);
    for(size_t i=0;i<h.size();++i)h[i]=__float2bfloat16_rn(float(int((i*17+salt)%31)-15)/32.f);
    CK(cudaMemcpy(p,h.data(),n,cudaMemcpyHostToDevice));return h;
  }
  std::vector<unsigned char> read(){std::vector<unsigned char> h(n);CK(cudaMemcpy(h.data(),p,n,cudaMemcpyDeviceToHost));return h;}
};
struct Kernel {
  CUmodule module; CUfunction function; unsigned grid,shared;
  Kernel(const char* path,const char* symbol,unsigned g,unsigned s):grid(g),shared(s){CK(cuModuleLoad(&module,path));CK(cuModuleGetFunction(&function,module,symbol));}
  ~Kernel(){CK(cuModuleUnload(module));}
  void launch(void**args){CK(cuLaunchKernel(function,grid,1,1,256,1,1,shared,nullptr,args,nullptr));}
  float time(void**args){
    for(int i=0;i<3;++i)launch(args);CK(cudaDeviceSynchronize());
    cudaEvent_t a,b;CK(cudaEventCreate(&a));CK(cudaEventCreate(&b));CK(cudaEventRecord(a));
    for(int i=0;i<20;++i)launch(args);
    CK(cudaEventRecord(b));CK(cudaEventSynchronize(b));float ms;CK(cudaEventElapsedTime(&ms,a,b));
    CK(cudaEventDestroy(a));CK(cudaEventDestroy(b));return ms*50;
  }
};
int main(int argc,char**argv){
  if(argc!=6&&argc!=7)return 1;
  const bool qkv=std::strcmp(argv[3],"qkv")==0,head=std::strcmp(argv[3],"head")==0;
  if(!qkv&&!head&&std::strcmp(argv[3],"output")!=0)return 1;
  const int tokens=std::atoi(argv[4]),rows=std::atoi(argv[5]);
  if(tokens<1||tokens>2048||rows<1||rows>32||tokens<rows)return 1;
  const int inner=qkv||head?1024:2048,columns=head?151936:qkv?4096:1024;
  CK(cudaSetDevice(0));CK(cudaFree(nullptr));
  Buffer counts(20),offsets(132),input(size_t(2048)*inner*2),w0(size_t(qkv?2048:columns)*inner*2),w1(size_t(qkv?1024:1)*inner*2),w2(size_t(qkv?1024:1)*inner*2),output(size_t(2048)*columns*2);
  int c[5]={rows,0,rows,tokens,rows},offset[33]={};
  for(int i=0;i<=rows;++i)offset[i]=i*tokens/rows;
  CK(cudaMemcpy(counts.p,c,20,cudaMemcpyHostToDevice));CK(cudaMemcpy(offsets.p,offset,132,cudaMemcpyHostToDevice));
  auto x=input.fill(3),a=w0.fill(7),b=w1.fill(11),d=w2.fill(13);
  const char* symbol=qkv?"lunaflux_luna_qkv_projection_bf16_release_v1":head?"lunaflux_luna_language_model_head_bf16_release_v1_segmented_greedy_v2":"lunaflux_luna_dense_projection_bf16_release_v1";
  Kernel old(argv[1],symbol,head?1187:qkv?4096:1024,head?8704:8192),now(argv[2],symbol,head?1187:qkv?4096:1024,head?8704:8192);
  void* qa[]={&counts.p,&input.p,&w0.p,&w1.p,&w2.p,&output.p};
  void* oa[]={&counts.p,&input.p,&w0.p,&output.p};
  void* ha[]={&counts.p,&offsets.p,&input.p,&w0.p,&output.p};
  void**args=qkv?qa:head?ha:oa;
  old.launch(args);CK(cudaDeviceSynchronize());auto expected=output.read();
  CK(cudaMemset(output.p,0,output.n));now.launch(args);CK(cudaDeviceSynchronize());auto actual=output.read();
  if(actual!=expected){std::fprintf(stderr,"bitwise mismatch\n");return 3;}
  for(int logical:{0,rows/2,rows-1}){
    const int row=head?offset[logical+1]-1:logical*tokens/rows;
    for(int col:{0,15,16,columns/2,columns-1}){
      auto* weight=&a;int wc=col;
      if(qkv&&col>=2048){weight=col<3072?&b:&d;wc=col-(col<3072?2048:3072);}
      float sum=0;for(int k=0;k<inner;++k)sum+=__bfloat162float(x[size_t(row)*inner+k])*__bfloat162float((*weight)[size_t(wc)*inner+k]);
      __nv_bfloat16 value;std::memcpy(&value,actual.data()+2*(size_t(row)*columns+col),2);
      const float ref=__bfloat162float(__float2bfloat16_rn(sum));
      if(!std::isfinite(__bfloat162float(value))||std::fabs(__bfloat162float(value)-ref)>0.02f){std::fprintf(stderr,"scalar mismatch row=%d col=%d\n",row,col);return 4;}
    }
  }
  std::puts("bitwise=true scalar_checks=passed");if(argc==7)return 0;
  for(int trial=0;trial<5;++trial){float a,b;if(trial%2){b=now.time(args);a=old.time(args);}else{a=old.time(args);b=now.time(args);}std::printf("trial=%d tokens=%d rows=%d old_us=%.6f new_us=%.6f\n",trial,tokens,rows,a,b);}
}
