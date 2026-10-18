#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <limits>
#include "scale.cuh"
static void ck(cudaError_t e) { if(e!=cudaSuccess){fprintf(stderr,"%s\n",cudaGetErrorString(e));exit(2);} }
template<class T> static T *alloc(size_t n) {T *p;ck(cudaMalloc(&p,n*sizeof(T)));return p;}
int main(int argc,char **argv) {
  bool sanitizer=argc>1;
  int *counts=alloc<int>(5),*error=alloc<int>(1);
  auto *x=alloc<__nv_bfloat16>((size_t)R*S*H);
  auto *old=alloc<unsigned char>(STORAGE),*next=alloc<unsigned char>(STORAGE);
  std::vector<__nv_bfloat16> host((size_t)R*S*H);
  std::vector<unsigned char> a(NUMERIC),b(NUMERIC);
  auto launch=[&](bool fresh){
    if(fresh){candidate_global<<<G,256>>>(counts,x,next,error);candidate_pack<<<(R*S*H+127)/128,128>>>(counts,x,next,error);}
    else {baseline_global<<<1,256>>>(counts,x,old,error);baseline_pack<<<(R*S*H+127)/128,128>>>(counts,x,old,error);}
    ck(cudaGetLastError());
  };
  std::vector<int> rows=sanitizer?std::vector<int>{1,3}:std::vector<int>{0,1,3,22,36,63,64,65};
  for(int live:rows) for(int variant=0;variant<(sanitizer?2:5);variant++) {
    for(size_t i=0;i<host.size();i++)host[i]=__float2bfloat16_rn(variant==1?0.f:float(int((i*7919)%131071)-65535)/32.f);
    if(variant==2)host[H/2]=__float2bfloat16_rn(std::numeric_limits<float>::quiet_NaN());
    if(variant==3)host[H/2]=__float2bfloat16_rn(std::numeric_limits<float>::infinity());
    if(variant==4){unsigned short hi=0x7f7f,lo=1;memcpy(&host[H/2],&hi,2);memcpy(&host[0],&lo,2);}
    int c[5]={0,0,0,live,0};ck(cudaMemcpy(counts,c,sizeof(c),cudaMemcpyHostToDevice));
    ck(cudaMemcpy(x,host.data(),host.size()*2,cudaMemcpyHostToDevice));
    int errors[2];
    for(int fresh=0;fresh<2;fresh++){
      ck(cudaMemset(fresh?next:old,0xa5,STORAGE));ck(cudaMemset(error,0,4));launch(fresh);
      ck(cudaMemcpy(errors+fresh,error,4,cudaMemcpyDeviceToHost));
      ck(cudaMemcpy(fresh?b.data():a.data(),fresh?next:old,NUMERIC,cudaMemcpyDeviceToHost));
    }
    if(a!=b||errors[0]!=errors[1]){size_t at=0;while(at<NUMERIC&&a[at]==b[at])at++;fprintf(stderr,"mismatch live=%d variant=%d offset=%zu errors=%d/%d\n",live,variant,at,errors[0],errors[1]);return 4;}
    printf("correctness live=%d variant=%d numeric_bytes=%zu error=%d bitwise=true\n",live,variant,NUMERIC,errors[0]);
  }
  if(!sanitizer){
    for(size_t i=0;i<host.size();i++)host[i]=__float2bfloat16_rn(float(int(i%8191)-4095)/32.f);
    ck(cudaMemcpy(x,host.data(),host.size()*2,cudaMemcpyHostToDevice));ck(cudaMemset(error,0,4));
    cudaStream_t stream;ck(cudaStreamCreate(&stream));
    cudaGraph_t graphs[2];cudaGraphExec_t execs[2];
    for(int fresh=0;fresh<2;fresh++){
      ck(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));
      for(int rep=0;rep<100;rep++){
        if(fresh){candidate_global<<<G,256,0,stream>>>(counts,x,next,error);candidate_pack<<<(R*S*H+127)/128,128,0,stream>>>(counts,x,next,error);}
        else{baseline_global<<<1,256,0,stream>>>(counts,x,old,error);baseline_pack<<<(R*S*H+127)/128,128,0,stream>>>(counts,x,old,error);}
      }
      ck(cudaStreamEndCapture(stream,graphs+fresh));ck(cudaGraphInstantiate(execs+fresh,graphs[fresh],nullptr,nullptr,0));
    }
    cudaEvent_t start,end;ck(cudaEventCreate(&start));ck(cudaEventCreate(&end));
    auto timed=[&](int fresh){ck(cudaEventRecord(start,stream));ck(cudaGraphLaunch(execs[fresh],stream));ck(cudaEventRecord(end,stream));ck(cudaEventSynchronize(end));float ms;ck(cudaEventElapsedTime(&ms,start,end));return ms*10.f;};
    for(int live:{1,3,22,36,64}){
      int c[5]={0,0,0,live,0};ck(cudaMemcpy(counts,c,sizeof(c),cudaMemcpyHostToDevice));timed(0);timed(1);
      for(int trial=0;trial<7;trial++){float old_us,next_us;if(trial%2){next_us=timed(1);old_us=timed(0);}else{old_us=timed(0);next_us=timed(1);}printf("timing rows=%d multiplier=%d groups=%d trial=%d old_us=%.6f new_us=%.6f\n",live,S,G,trial,old_us,next_us);}
    }
    ck(cudaEventDestroy(start));ck(cudaEventDestroy(end));for(int i=0;i<2;i++){ck(cudaGraphExecDestroy(execs[i]));ck(cudaGraphDestroy(graphs[i]));}ck(cudaStreamDestroy(stream));
  }
  ck(cudaFree(counts));ck(cudaFree(error));ck(cudaFree(x));ck(cudaFree(old));ck(cudaFree(next));ck(cudaDeviceReset());puts("released=true");return 0;
}
