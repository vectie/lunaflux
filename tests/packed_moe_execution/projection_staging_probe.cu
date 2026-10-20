#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "prefill.cuh"

#if defined(LUNA_PROJECTION_BF16_OUTPUT)
using Output = __nv_bfloat16;
static double output_value(Output x) { return __bfloat162float(x); }
#else
using Output = float;
static double output_value(Output x) { return x; }
#endif

static void ck(cudaError_t e) {
  if (e != cudaSuccess) { fprintf(stderr, "%s\n", cudaGetErrorString(e)); exit(2); }
}
template<class T> static T *allocate(size_t n) {
  T *p; ck(cudaMalloc(&p, n * sizeof(T))); return p;
}
int main(int argc, char **argv) {
  if (argc != 5) return 2;
  const int shape=atoi(argv[1]), rows=atoi(argv[2]), candidate=atoi(argv[3]);
  const bool profile=strcmp(argv[4],"profile")==0;
  const bool sanitize=strcmp(argv[4],"sanitize")==0;
  if(shape<0 || shape>=int(sizeof(widths)/sizeof(widths[0])) || rows<0 || rows>65 || candidate<0 || candidate>=4) return 2;
  const int n=widths[shape], k=inners[shape];
  // Preserve the actual capacity/decode-reserved launch domain, including
  // inactive prefill tiles. Do not compare a reduced-grid probe to serving.
  const int grid=std::max((n+3)/4,4*((n+63)/64));
  auto *counts=allocate<int>(4);
  auto *x=allocate<__nv_bfloat16>(64*k), *w=allocate<__nv_bfloat16>((size_t)n*k);
  auto *y=allocate<Output>(64*n);
  std::vector<__nv_bfloat16> hx(64*k),hw((size_t)n*k);
  unsigned seed=713;
  for(auto *values:{&hx,&hw}) for(auto &v:*values) {
    seed=1664525u*seed+1013904223u;
    v=__float2bfloat16_rn(float(int(seed>>16)-32768)*0.000001f);
  }
  int c[]={0,1,1,rows};
  ck(cudaMemcpy(counts,c,sizeof(c),cudaMemcpyHostToDevice));
  ck(cudaMemcpy(x,hx.data(),hx.size()*2,cudaMemcpyHostToDevice));
  ck(cudaMemcpy(w,hw.data(),hw.size()*2,cudaMemcpyHostToDevice));
  void *args[]={&counts,&x,&w,&y};
  auto launch=[&](int v,cudaStream_t stream=0) {
    ck(cudaLaunchKernel(functions[shape][v],dim3(grid),dim3(128),args,0,stream));
  };
  if(profile) { launch(candidate); ck(cudaDeviceSynchronize()); }
  else {
    std::vector<Output> old(64*n),fresh(64*n);
    ck(cudaMemset(y,0x55,old.size()*sizeof(Output))); launch(0);
    ck(cudaMemcpy(old.data(),y,old.size()*sizeof(Output),cudaMemcpyDeviceToHost));
    ck(cudaMemset(y,0x55,fresh.size()*sizeof(Output))); launch(candidate);
    ck(cudaMemcpy(fresh.data(),y,fresh.size()*sizeof(Output),cudaMemcpyDeviceToHost));
    bool exact=memcmp(old.data(),fresh.data(),old.size()*sizeof(Output))==0;
    if(!exact) { fprintf(stderr,"different output/inactive bytes\n"); return 4; }
    double oracle=0;
    if(rows>0 && rows<=64) for(int r:{0,rows-1}) for(int col:{0,n/2,n-1}) {
      double sum=0;
      for(int j=0;j<k;j++) sum+=double(__bfloat162float(hx[r*k+j]))*__bfloat162float(hw[(size_t)col*k+j]);
      oracle=std::max(oracle,std::abs(output_value(fresh[r*n+col])-sum));
    }
    if(!(oracle<=0.003)) return 5;
    printf("correctness=passed shape=%d rows=%d candidate=%d grid=%d block=128 bitwise=true oracle_maxabs=%.12g\n",shape,rows,candidate,grid,oracle);
    if(!sanitize && rows>0 && rows<=64) {
      cudaStream_t stream; ck(cudaStreamCreate(&stream));
      cudaGraph_t graphs[2]; cudaGraphExec_t executions[2];
      for(int v=0;v<2;v++) {
        ck(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));
        for(int i=0;i<20;i++) launch(v?candidate:0,stream);
        ck(cudaStreamEndCapture(stream,&graphs[v]));
        ck(cudaGraphInstantiate(&executions[v],graphs[v],nullptr,nullptr,0));
        ck(cudaGraphLaunch(executions[v],stream));
      }
      ck(cudaStreamSynchronize(stream));
      cudaEvent_t a,b; ck(cudaEventCreate(&a)); ck(cudaEventCreate(&b));
      for(int trial=0;trial<5;trial++) {
        float times[2];
        for(int order=0;order<2;order++) {
          int v=trial%2?1-order:order;
          ck(cudaEventRecord(a,stream)); ck(cudaGraphLaunch(executions[v],stream));
          ck(cudaEventRecord(b,stream)); ck(cudaEventSynchronize(b));
          float ms; ck(cudaEventElapsedTime(&ms,a,b)); times[v]=ms*50.f;
        }
        printf("tokens=%d rows=%d trial=%d old_us=%.9g new_us=%.9g bitwise=true maxabs=0 oracle_maxabs=%.12g\n",rows,rows,trial,times[0],times[1],oracle);
      }
      ck(cudaEventDestroy(a)); ck(cudaEventDestroy(b));
      for(int i=0;i<2;i++) { ck(cudaGraphExecDestroy(executions[i])); ck(cudaGraphDestroy(graphs[i])); }
      ck(cudaStreamDestroy(stream));
    }
  }
  ck(cudaFree(counts)); ck(cudaFree(x)); ck(cudaFree(w)); ck(cudaFree(y));
  ck(cudaDeviceReset()); puts("released=true");
}
