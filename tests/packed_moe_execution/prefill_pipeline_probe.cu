#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cstring>
#include "prefill.cuh"
static void ck(cudaError_t e){if(e!=cudaSuccess){fprintf(stderr,"%s\n",cudaGetErrorString(e));exit(2);}}
template<class T>static T *alloc(size_t n){T*p;ck(cudaMalloc(&p,n*sizeof(T)));return p;}
int main(int argc,char **argv){
  bool sanitize=argc>1;int chosen=argc>2?atoi(argv[2]):3;
  for(int shape=0;shape<4;shape++){
    int n=widths[shape],k=inners[shape],grid=std::max(n/4,4*((n+63)/64));
    auto *c=alloc<int>(4);auto *x=alloc<__nv_bfloat16>(64*k),*w=alloc<__nv_bfloat16>((size_t)n*k);auto *y=alloc<float>(64*n);
    std::vector<__nv_bfloat16>hx(64*k),hw((size_t)n*k);unsigned state=713;
    for(auto *v:{&hx,&hw})for(auto &b:*v){state=state*1664525u+1013904223u;b=__float2bfloat16_rn(float(int(state>>16)-32768)*.000031f);}
    ck(cudaMemcpy(x,hx.data(),hx.size()*2,cudaMemcpyHostToDevice));ck(cudaMemcpy(w,hw.data(),hw.size()*2,cudaMemcpyHostToDevice));
    void *args[]={&c,&x,&w,&y};
    auto run=[&](int variant,cudaStream_t stream=0){ck(cudaLaunchKernel(functions[shape][variant],dim3(grid),dim3(128),args,0,stream));};
    for(int rows:(sanitize?std::vector<int>{1,22}:std::vector<int>{0,1,3,22,36,64,65})){
      int counts[]={0,1,1,rows};ck(cudaMemcpy(c,counts,16,cudaMemcpyHostToDevice));std::vector<unsigned>a(64*n),b(64*n);
      for(int variant=0;variant<4;variant++){
        if(sanitize&&variant!=0&&variant!=chosen)continue;
        ck(cudaMemset(y,0x55,64*n*4));run(variant);ck(cudaMemcpy(b.data(),y,b.size()*4,cudaMemcpyDeviceToHost));
        if(variant==0)a=b;else if(a!=b){size_t at=0;while(at<a.size()&&a[at]==b[at])at++;fprintf(stderr,"mismatch shape=%d rows=%d variant=%d cell=%zu %x/%x\n",shape,rows,variant,at,a[at],b[at]);return 4;}
      }
      printf("correct shape=%d rows=%d f32_bitwise=true inactive_preserved=true\n",shape,rows);
      if(sanitize||rows<1||rows>64)continue;
      cudaStream_t stream;ck(cudaStreamCreate(&stream));cudaGraph_t graphs[4];cudaGraphExec_t exec[4];
      for(int variant=0;variant<4;variant++){ck(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));for(int i=0;i<20;i++)run(variant,stream);ck(cudaStreamEndCapture(stream,graphs+variant));ck(cudaGraphInstantiate(exec+variant,graphs[variant],nullptr,nullptr,0));ck(cudaGraphLaunch(exec[variant],stream));}ck(cudaStreamSynchronize(stream));
      cudaEvent_t first,last;ck(cudaEventCreate(&first));ck(cudaEventCreate(&last));
      for(int trial=0;trial<7;trial++)for(int order=0;order<4;order++){int variant=trial%2?3-order:order;ck(cudaEventRecord(first,stream));ck(cudaGraphLaunch(exec[variant],stream));ck(cudaEventRecord(last,stream));ck(cudaEventSynchronize(last));float ms;ck(cudaEventElapsedTime(&ms,first,last));printf("timing shape=%d rows=%d variant=%d trial=%d us=%.6f\n",shape,rows,variant,trial,ms*50.f);}
      ck(cudaEventDestroy(first));ck(cudaEventDestroy(last));for(int i=0;i<4;i++){ck(cudaGraphExecDestroy(exec[i]));ck(cudaGraphDestroy(graphs[i]));}ck(cudaStreamDestroy(stream));
    }
    ck(cudaFree(c));ck(cudaFree(x));ck(cudaFree(w));ck(cudaFree(y));
  }
  ck(cudaDeviceReset());puts("released=true");return 0;
}
