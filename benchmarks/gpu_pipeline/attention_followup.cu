#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
#include <cmath>
#include <cstring>
#define CK(x) do { auto e=(x); if(e!=0){fprintf(stderr,"CUDA %d line %d\n",int(e),__LINE__);exit(2);} }while(0)
#include "attention_referee.h"
struct Buffer {
 void*p=nullptr; size_t n;
 Buffer(size_t bytes):n(bytes){CK(cudaMalloc(&p,n));CK(cudaMemset(p,0,n));}
 ~Buffer(){CK(cudaFree(p));}
 void ints(const std::vector<int>&v){if(v.size()*4>n)exit(3);CK(cudaMemcpy(p,v.data(),v.size()*4,cudaMemcpyHostToDevice));}
 void fill(int salt){std::vector<__nv_bfloat16>v(n/2);for(size_t i=0;i<v.size();i++)v[i]=__float2bfloat16(float(int((i*17+salt)%31)-15)/32.f);CK(cudaMemcpy(p,v.data(),n,cudaMemcpyHostToDevice));}
 std::vector<unsigned char> read(){std::vector<unsigned char>v(n);CK(cudaMemcpy(v.data(),p,n,cudaMemcpyDeviceToHost));return v;}
};
struct Kernel {
 CUmodule m;CUfunction f;unsigned x,y,b,s;
 Kernel(const char*path,const char*symbol,unsigned gx,unsigned gy,unsigned block,unsigned shared):x(gx),y(gy),b(block),s(shared){CK(cuModuleLoad(&m,path));CK(cuModuleGetFunction(&f,m,symbol));if(s)CK(cuFuncSetAttribute(f,CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,s));int r;CK(cuFuncGetAttribute(&r,CU_FUNC_ATTRIBUTE_NUM_REGS,f));printf("module=%s registers=%d\n",path,r);}
 ~Kernel(){CK(cuModuleUnload(m));}
 void launch(void**args){CK(cuLaunchKernel(f,x,y,1,b,1,1,s,nullptr,args,nullptr));}
 float time(void**args){cudaEvent_t a,b;CK(cudaEventCreate(&a));CK(cudaEventCreate(&b));for(int i=0;i<3;i++)launch(args);CK(cudaDeviceSynchronize());CK(cudaEventRecord(a));for(int i=0;i<30;i++)launch(args);CK(cudaEventRecord(b));CK(cudaEventSynchronize(b));float ms;CK(cudaEventElapsedTime(&ms,a,b));CK(cudaEventDestroy(a));CK(cudaEventDestroy(b));return ms*1000/30;}
};
int main(int argc,char**argv){
 // KIND OLD NEW OLD_GX OLD_GY OLD_SHARED NEW_GX NEW_GY NEW_SHARED MODE TOKENS ROWS
 if(argc!=16)return 1;int past=atoi(argv[14]);if(past<0||past>8192)return 1;const bool ingress=std::string(argv[1])=="ingress";int mode=atoi(argv[10]),tokens=atoi(argv[11]),rows=atoi(argv[12]);
 if(tokens<1||tokens>2048||rows<1||rows>32||tokens<rows)return 1;
 CK(cudaSetDevice(0));CK(cudaFree(nullptr));
 Buffer counts(20),positions(2048*4),offsets(33*4),seq(32*4),po(33*4),pi(8192*4);
 Buffer input(2048*4096*2),output(2048*(ingress?4096:2048)*2),keys(8192*8*1024*2),values(8192*8*1024*2),qw(256),kw(256);
 input.fill(3);qw.fill(17);kw.fill(19);keys.fill(29);values.fill(31);
 std::vector<int> off{0},pages{0},indices,pos,lengths;int token=0;
 for(int r=0;r<rows;r++){int n=tokens/rows+(r<tokens%rows);int context=n+past+((rows==1)?0:r%3*8);lengths.push_back(context);for(int i=0;i<n;i++)pos.push_back(context-n+i);token+=n;off.push_back(token);int np=(context+7)/8;for(int i=0;i<np;i++)indices.push_back(int(indices.size()));pages.push_back(int(indices.size()));}
 counts.ints({rows,0,rows,tokens,int(indices.size())});positions.ints(pos);offsets.ints(off);seq.ints(lengths);po.ints(pages);pi.ints(indices);
 const char* symbol=ingress?"lunaflux_qwen_qknorm_rope_kvwrite_bf16_head128_production_v2":"lunaflux_attention_prefill_tile_compiler_v1";
 Kernel old(argv[2],symbol,atoi(argv[4]),atoi(argv[5]),ingress?128:256,atoi(argv[6]));
 Kernel now(argv[3],argv[13],atoi(argv[7]),atoi(argv[8]),atoi(argv[15]),atoi(argv[9]));
 void*a[]={&counts.p,&positions.p,&offsets.p,&seq.p,&po.p,&pi.p,&input.p,&output.p,&keys.p,&values.p};
 void*q[]={&counts.p,&positions.p,&offsets.p,&seq.p,&po.p,&pi.p,&input.p,&qw.p,&kw.p,&output.p,&keys.p,&values.p};
 void**args=ingress?q:a;
 if(mode){(mode==1?old:now).launch(args);CK(cudaDeviceSynchronize());return 0;}
 CK(cudaMemset(output.p,0xa5,output.n));old.launch(args);CK(cudaDeviceSynchronize());auto expected=output.read();auto ek=keys.read(),ev=values.read();
 CK(cudaMemset(output.p,0xa5,output.n));if(ingress){keys.fill(29);values.fill(31);}now.launch(args);CK(cudaDeviceSynchronize());
 auto actual=output.read();bool bitwise=expected==actual;float maxabs=0;
 if(!ingress){for(size_t i=0;i<size_t(tokens)*2048;i++){__nv_bfloat16 a,b;std::memcpy(&a,expected.data()+i*2,2);std::memcpy(&b,actual.data()+i*2,2);float x=__bfloat162float(a),y=__bfloat162float(b);if(!std::isfinite(x)||!std::isfinite(y))return 4;maxabs=fmaxf(maxabs,fabsf(x-y));}}
 if((ingress&&!bitwise)||maxabs>0.003f||ek!=keys.read()||ev!=values.read()){fprintf(stderr,"numeric mismatch abs=%g\n",maxabs);return 4;}
 check_attention_referee(input.read(),ek,ev,actual,pos,off,pages,indices);
 for(int trial=0;trial<5;trial++){float a,b;if(trial%2){b=now.time(args);a=old.time(args);}else{a=old.time(args);b=now.time(args);}printf("tokens=%d rows=%d trial=%d old_us=%.6f new_us=%.6f bitwise=%s maxabs=%g\n",tokens,rows,trial,a,b,bitwise?"true":"false",maxabs);}
}
