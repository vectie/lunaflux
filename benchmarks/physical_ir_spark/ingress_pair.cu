// Diagnostic driver harness for the production full-ingress source. No model
// server or production runtime is linked into this bounded kernel benchmark.
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include "ingress_shape.h"
#define CK(call) do { auto e=(call); if(e){ std::fprintf(stderr,"CUDA error=%d line=%d\n",int(e),__LINE__); std::exit(2); } } while(0)

struct Buffer {
  void *p; size_t size;
  explicit Buffer(size_t bytes):size(bytes){CK(cudaMalloc(&p,bytes));zero();}
  ~Buffer(){CK(cudaFree(p));}
  void zero(){CK(cudaMemset(p,0,size));}
  void put(const void* data){CK(cudaMemcpy(p,data,size,cudaMemcpyHostToDevice));}
  std::vector<__nv_bfloat16> read(){std::vector<__nv_bfloat16> result(size/2);CK(cudaMemcpy(result.data(),p,size,cudaMemcpyDeviceToHost));return result;}
};

struct Kernel {
  CUmodule module; CUfunction function;
  explicit Kernel(const char* source){
    CK(cuModuleLoad(&module,source));
    CK(cuModuleGetFunction(&function,module,"lunaflux_fused_qwen_qkv_qknorm_rope_kvwrite_bf16_head128_production_v2"));
  }
  ~Kernel(){CK(cuModuleUnload(module));}
  void launch(int tokens,void**args){CK(cuLaunchKernel(function,(tokens+15)/16,LF_PACKED_HEADS,1,128,1,1,0,nullptr,args,nullptr));}
  float time(int tokens,void**args){
    for(int i=0;i<10;++i)launch(tokens,args);
    CK(cudaDeviceSynchronize());
    cudaEvent_t start,finish;CK(cudaEventCreate(&start));CK(cudaEventCreate(&finish));
    CK(cudaEventRecord(start));for(int i=0;i<100;++i)launch(tokens,args);
    CK(cudaEventRecord(finish));CK(cudaEventSynchronize(finish));
    float ms;CK(cudaEventElapsedTime(&ms,start,finish));
    CK(cudaEventDestroy(start));CK(cudaEventDestroy(finish));return ms*10;
  }
};

int main(int argc,char**argv){
  if(argc!=3)return 1;
  CK(cudaSetDevice(0));CK(cudaFree(nullptr));
  Kernel before(argv[1]),now(argv[2]);
  for(int tokens:{1,2,7,8,15,16,17,31,32}){
    int counts[5]={1,0,1,tokens,2},offsets[2]={0,tokens},tables[2]={0,2},pages[2]={0,1};
    std::vector<int> positions(tokens);for(int i=0;i<tokens;++i)positions[i]=i;
    Buffer dc(sizeof counts),dpos(tokens*4),doff(8),dseq(4),dtoff(8),dpage(8);
    dc.put(counts);dpos.put(positions.data());doff.put(offsets);dseq.put(&tokens);dtoff.put(tables);dpage.put(pages);
    std::vector<__nv_bfloat16> x(tokens*LF_INPUT_WIDTH),w(LF_QUERY_HEADS*LF_HEAD_DIM*LF_INPUT_WIDTH,__float2bfloat16_rn(1.0f/LF_INPUT_WIDTH)),norm(LF_HEAD_DIM,__float2bfloat16_rn(1.0f));
    for(int t=0;t<tokens;++t)for(int k=0;k<LF_INPUT_WIDTH;++k)x[t*LF_INPUT_WIDTH+k]=__float2bfloat16_rn(float(t+1)/8);
    Buffer dx(x.size()*2),dw(w.size()*2),dn(norm.size()*2),dout(tokens*LF_OUTPUT_WIDTH*2);
    Buffer dk(LF_TOTAL_PAGE_COUNT*LF_PAGE_STRIDE_BF16*2),dv(LF_TOTAL_PAGE_COUNT*LF_PAGE_STRIDE_BF16*2);
    dx.put(x.data());dw.put(w.data());dn.put(norm.data());
    void*args[]={&dc.p,&dpos.p,&doff.p,&dseq.p,&dtoff.p,&dpage.p,&dx.p,&dw.p,&dw.p,&dw.p,&dn.p,&dn.p,&dout.p,&dk.p,&dv.p};
    before.launch(tokens,args);CK(cudaDeviceSynchronize());auto expected=dout.read(),expected_k=dk.read(),expected_v=dv.read();
    dout.zero();dk.zero();dv.zero();now.launch(tokens,args);CK(cudaDeviceSynchronize());auto actual=dout.read(),key=dk.read(),value=dv.read();
    if(std::memcmp(expected.data(),actual.data(),dout.size)||std::memcmp(expected_k.data(),key.data(),dk.size)||std::memcmp(expected_v.data(),value.data(),dv.size)){std::fprintf(stderr,"ingress bitwise mismatch dimension=%d tokens=%d\n",LF_HEAD_DIM,tokens);return 3;}
    for(int t=0;t<tokens;++t)for(int h=0;h<LF_PACKED_HEADS;++h)for(int c=0;c<LF_HEAD_DIM;++c){
      float v=float(t+1)/8,ref=v;
      if(h<LF_QK_HEADS){
        float n=__bfloat162float(__float2bfloat16_rn(v/std::sqrt(v*v+LF_NORM_EPSILON)));
        int pair=c%(LF_HEAD_DIM/2);float angle=float(t)*float(std::pow(1000000.0,-2.0*pair/LF_HEAD_DIM));
        ref=__bfloat162float(__float2bfloat16_rn(n*(std::cos(angle)+(c<LF_HEAD_DIM/2?-1:1)*std::sin(angle))));
      }
      float got=__bfloat162float(actual[t*LF_OUTPUT_WIDTH+h*LF_HEAD_DIM+c]);
      if(!std::isfinite(got)||std::fabs(got-ref)>0.02f){std::fprintf(stderr,"ingress scalar mismatch\n");return 4;}
      if(h>=LF_QUERY_HEADS){
        int kh=h<LF_QK_HEADS?h-LF_QUERY_HEADS:h-LF_QK_HEADS;
        size_t index=size_t(t/LF_TOKENS_PER_PAGE)*LF_PAGE_STRIDE_BF16+(t%LF_TOKENS_PER_PAGE)*LF_KV_WIDTH+kh*LF_HEAD_DIM+c;
        if(__bfloat162float((h<LF_QK_HEADS?key:value)[index])!=got)return 5;
      }
    }
    for(int trial=0;trial<3;++trial){
      float a,b;if(trial%2){b=now.time(tokens,args);a=before.time(tokens,args);}else{a=before.time(tokens,args);b=now.time(tokens,args);}
      std::printf("dimension=%d tokens=%d trial=%d old_us=%.6f new_us=%.6f bitwise=true scalar=passed kv=passed\n",LF_HEAD_DIM,tokens,trial,a,b);
    }
  }
}
