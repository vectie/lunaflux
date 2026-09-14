#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#define CK(x) do {auto error=(x);if(error){std::fprintf(stderr,"CUDA %d line %d\n",int(error),__LINE__);std::exit(2);}}while(0)
struct Buffer {
  void* p=nullptr; size_t n;
  Buffer(size_t size):n(size){CK(cudaMalloc(&p,n));CK(cudaMemset(p,0,n));}
  ~Buffer(){CK(cudaFree(p));}
  std::vector<__nv_bfloat16> fill(int salt){
    std::vector<__nv_bfloat16> h(n/2);
    for(size_t i=0;i<h.size();++i)h[i]=__float2bfloat16_rn(float(int((i*17+salt)%31)-15)/32.f);
    CK(cudaMemcpy(p,h.data(),n,cudaMemcpyHostToDevice));return h;
  }
  std::vector<unsigned char> read(){std::vector<unsigned char> h(n);CK(cudaMemcpy(h.data(),p,n,cudaMemcpyDeviceToHost));return h;}
};
struct Pipeline {
  CUmodule module; CUfunction gate,down;
  Pipeline(const char* path){
    CK(cuModuleLoad(&module,path));
    CK(cuModuleGetFunction(&gate,module,"lunaflux_luna_gated_mlp_bf16_release_v1"));
    CK(cuModuleGetFunction(&down,module,"lunaflux_luna_gated_mlp_bf16_release_v1_down"));
  }
  ~Pipeline(){CK(cuModuleUnload(module));}
  void launch(void** args,int part){
    if(part!=2)CK(cuLaunchKernel(gate,1536,1,1,512,1,1,16384,nullptr,args,nullptr));
    if(part!=1)CK(cuLaunchKernel(down,2048,1,1,128,1,1,4096,nullptr,args,nullptr));
  }
  float time(void** args,int part){
    cudaEvent_t a,b;CK(cudaEventCreate(&a));CK(cudaEventCreate(&b));
    for(int i=0;i<3;++i)launch(args,part);CK(cudaDeviceSynchronize());
    CK(cudaEventRecord(a));for(int i=0;i<20;++i)launch(args,part);
    CK(cudaEventRecord(b));CK(cudaEventSynchronize(b));float ms;CK(cudaEventElapsedTime(&ms,a,b));
    CK(cudaEventDestroy(a));CK(cudaEventDestroy(b));return ms*1000/20;
  }
};
int main(int argc,char**argv){
  if(argc!=4&&argc!=5)return 1;int tokens=std::atoi(argv[3]);if(tokens<1||tokens>2048)return 1;
  CK(cudaSetDevice(0));CK(cudaFree(nullptr));
  Buffer counts(20),input(2048*1024*2),gate(3072*1024*2),up(3072*1024*2),down(3072*1024*2),output(2048*1024*2),workspace(2048*3072*2);
  int c[5]={1,0,1,tokens,1};CK(cudaMemcpy(counts.p,c,sizeof(c),cudaMemcpyHostToDevice));
  auto x=input.fill(3),g=gate.fill(7),u=up.fill(11),d=down.fill(13);
  Pipeline old(argv[1]),now(argv[2]);void*args[]={&counts.p,&input.p,&gate.p,&up.p,&down.p,&output.p,&workspace.p};
  old.launch(args,0);CK(cudaDeviceSynchronize());auto expected=output.read(),expected_gate=workspace.read();
  CK(cudaMemset(output.p,0,output.n));CK(cudaMemset(workspace.p,0,workspace.n));
  now.launch(args,0);CK(cudaDeviceSynchronize());auto actual=output.read(),actual_gate=workspace.read();
  if(expected!=actual||expected_gate!=actual_gate){std::fprintf(stderr,"ordered pipeline mismatch\n");return 3;}
  for(int row : {0,tokens/2,tokens-1}) {
    for(int col : {0,15,16,1023,2048,3071}) {
      float a=0,b=0;for(int k=0;k<1024;++k){a+=__bfloat162float(x[row*1024+k])*__bfloat162float(g[col*1024+k]);b+=__bfloat162float(x[row*1024+k])*__bfloat162float(u[col*1024+k]);}
      float ref=__bfloat162float(__float2bfloat16_rn(a/(1+std::exp(-a))*b));__nv_bfloat16 value;
      std::memcpy(&value,actual_gate.data()+2*(row*3072+col),2);
      if(!std::isfinite(ref)||std::fabs(__bfloat162float(value)-ref)>0.02f)return 4;
    }
    for(int col : {0,15,16,511,1023}) {
      float sum=0;for(int k=0;k<3072;++k){__nv_bfloat16 value;std::memcpy(&value,actual_gate.data()+2*(row*3072+k),2);sum+=__bfloat162float(value)*__bfloat162float(d[col*3072+k]);}
      __nv_bfloat16 value;std::memcpy(&value,actual.data()+2*(row*1024+col),2);
      float ref=__bfloat162float(__float2bfloat16_rn(sum));
      if(!std::isfinite(ref)||std::fabs(__bfloat162float(value)-ref)>0.02f)return 5;
    }
  }
  std::puts("bitwise_pipeline=true scalar_checks=passed");
  if(argc==5)return 0;
  for(int part : {1,2,0})for(int trial=0;trial<5;++trial){
    float a,b;if(trial%2){b=now.time(args,part);a=old.time(args,part);}else{a=old.time(args,part);b=now.time(args,part);}
    std::printf("tokens=%d part=%d trial=%d old_us=%.6f new_us=%.6f\n",tokens,part,trial,a,b);
  }
}
