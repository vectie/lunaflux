#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#define CK(x) do {auto e=(x);if(e){std::fprintf(stderr,"CUDA %d at %d\n",int(e),__LINE__);std::exit(2);}}while(0)
struct Buffer {
 void *p=nullptr; size_t n;
 Buffer(size_t bytes):n(bytes){CK(cudaMalloc(&p,n));CK(cudaMemset(p,0,n));}
 ~Buffer(){CK(cudaFree(p));}
 std::vector<__nv_bfloat16> fill(int salt){std::vector<__nv_bfloat16> h(n/2);for(size_t i=0;i<h.size();++i)h[i]=__float2bfloat16_rn(float(int((i*17+salt)%31)-15)/32.f);CK(cudaMemcpy(p,h.data(),n,cudaMemcpyHostToDevice));return h;}
 std::vector<unsigned char> read(){std::vector<unsigned char> h(n);CK(cudaMemcpy(h.data(),p,n,cudaMemcpyDeviceToHost));return h;}
};
struct Kernel {
 CUmodule m; CUfunction f;
 Kernel(const char* path){CK(cuModuleLoad(&m,path));CK(cuModuleGetFunction(&f,m,"lunaflux_luna_gated_mlp_bf16_release_v1"));int regs;CK(cuFuncGetAttribute(&regs,CU_FUNC_ATTRIBUTE_NUM_REGS,f));std::printf("module=%s registers=%d\n",path,regs);}
 ~Kernel(){CK(cuModuleUnload(m));}
 void launch(void** args){CK(cuLaunchKernel(f,1536,1,1,512,1,1,16384,nullptr,args,nullptr));}
 float time(void** args){cudaEvent_t a,b;CK(cudaEventCreate(&a));CK(cudaEventCreate(&b));for(int i=0;i<3;i++)launch(args);CK(cudaDeviceSynchronize());CK(cudaEventRecord(a));for(int i=0;i<30;i++)launch(args);CK(cudaEventRecord(b));CK(cudaEventSynchronize(b));float ms;CK(cudaEventElapsedTime(&ms,a,b));CK(cudaEventDestroy(a));CK(cudaEventDestroy(b));return ms*1000/30;}
};
int main(int argc,char**argv){
 if(argc!=5)return 1;int tokens=std::atoi(argv[3]),mode=std::atoi(argv[4]);if(tokens<1||tokens>2048)return 1;
 CK(cudaSetDevice(0));CK(cudaFree(nullptr));
 Buffer counts(20),input(2048*1024*2),gate(3072*1024*2),up(3072*1024*2),down(3072*1024*2),output(2048*1024*2),workspace(2048*3072*2);
 int c[5]={1,0,1,tokens,1};CK(cudaMemcpy(counts.p,c,sizeof(c),cudaMemcpyHostToDevice));
 auto x=input.fill(3),g=gate.fill(7),u=up.fill(11);down.fill(13);
 Kernel old(argv[1]),now(argv[2]);void*args[]={&counts.p,&input.p,&gate.p,&up.p,&down.p,&output.p,&workspace.p};
 if(mode){now.launch(args);CK(cudaDeviceSynchronize());return 0;}
 CK(cudaMemset(workspace.p,0xa5,workspace.n));old.launch(args);CK(cudaDeviceSynchronize());auto expected=workspace.read();
 CK(cudaMemset(workspace.p,0xa5,workspace.n));now.launch(args);CK(cudaDeviceSynchronize());auto actual=workspace.read();
 if(expected!=actual){std::fprintf(stderr,"ordered GEMM output changed\n");return 3;}
 float maxerror=0;
 for(int row : {0,tokens/2,tokens-1})for(int col : {0,15,16,1023,2048,3071}){
   float a=0,b=0;for(int k=0;k<1024;k++){a+=__bfloat162float(x[row*1024+k])*__bfloat162float(g[col*1024+k]);b+=__bfloat162float(x[row*1024+k])*__bfloat162float(u[col*1024+k]);}
   const float ref=__bfloat162float(__float2bfloat16_rn(a/(1+std::exp(-a))*b));__nv_bfloat16 value;std::memcpy(&value,actual.data()+2*(row*3072+col),2);
   float error=std::fabs(__bfloat162float(value)-ref);maxerror=std::fmax(maxerror,error);if(!std::isfinite(error)||error>0.02f)return 4;
 }
 for(int t=0;t<5;t++){float a,b;if(t%2){b=now.time(args);a=old.time(args);}else{a=old.time(args);b=now.time(args);}std::printf("tokens=%d trial=%d old_us=%.6f new_us=%.6f bitwise=true scalar_maxabs=%g\n",tokens,t,a,b,maxerror);}
}
