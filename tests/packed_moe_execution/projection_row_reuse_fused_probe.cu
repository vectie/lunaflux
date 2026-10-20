#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "prefill.cuh"
static void ck(cudaError_t e) {if(e!=cudaSuccess){fprintf(stderr,"%s\n",cudaGetErrorString(e));exit(2);}}
template<class T> static T *alloc(size_t n) {T*p;ck(cudaMalloc(&p,n*sizeof(T)));return p;}
int main(int argc,char**argv) {
  if(argc!=5)return 2;
  int s=atoi(argv[1]),rows=atoi(argv[2]),v=atoi(argv[3]);
  if(s<0||s>1||rows<0||rows>65||v<0||v>3)return 2;
  bool profile=strcmp(argv[4],"profile")==0,sanitize=strcmp(argv[4],"sanitize")==0;
  int k=s?33:4096;const int *width=s?widths1:widths0,*grid=s?grids1:grids0;
  auto*c=alloc<int>(4);auto*x=alloc<__nv_bfloat16>(64*k);
  __nv_bfloat16 *w[6],*y[6];std::vector<__nv_bfloat16> hx(64*k),hw[6];
  unsigned seed=713;auto fill=[&](auto &a){for(auto &z:a){seed=1664525u*seed+1013904223u;z=__float2bfloat16_rn(float(int(seed>>16)-32768)*.000001f);}};
  fill(hx);ck(cudaMemcpy(x,hx.data(),hx.size()*2,cudaMemcpyHostToDevice));
  for(int i=0;i<6;i++){w[i]=alloc<__nv_bfloat16>(size_t(width[i])*k);y[i]=alloc<__nv_bfloat16>(64*width[i]);hw[i].resize(size_t(width[i])*k);fill(hw[i]);ck(cudaMemcpy(w[i],hw[i].data(),hw[i].size()*2,cudaMemcpyHostToDevice));}
  int counts[]={0,1,1,rows};ck(cudaMemcpy(c,counts,sizeof(counts),cudaMemcpyHostToDevice));
  void *args[]={&c,&x,&w[0],&y[0],&w[1],&y[1],&w[2],&y[2],&w[3],&y[3],&w[4],&y[4],&w[5],&y[5]};
  auto run=[&](int which,cudaStream_t stream=0){ck(cudaLaunchKernel(functions[s][which],dim3(grid[which]),dim3(128),args,0,stream));};
  if(profile){run(v);ck(cudaDeviceSynchronize());}else{
    std::vector<unsigned short> before[6],after[6];
    for(int i=0;i<6;i++){before[i].resize(64*width[i]);after[i].resize(64*width[i]);ck(cudaMemset(y[i],0x55,before[i].size()*2));}
    run(0);
    for(int i=0;i<6;i++){ck(cudaMemcpy(before[i].data(),y[i],before[i].size()*2,cudaMemcpyDeviceToHost));ck(cudaMemset(y[i],0x55,before[i].size()*2));}
    run(v);double maxerr=0;
    for(int i=0;i<6;i++){
      ck(cudaMemcpy(after[i].data(),y[i],after[i].size()*2,cudaMemcpyDeviceToHost));
      if(before[i]!=after[i]){fprintf(stderr,"mismatch segment%d\n",i);return 4;}
      if(rows>0&&rows<=64)for(int row:{0,rows-1})for(int col:{0,width[i]/2,width[i]-1}){
        double sum=0;for(int z=0;z<k;z++)sum+=double(__bfloat162float(hx[row*k+z]))*__bfloat162float(hw[i][size_t(col)*k+z]);
        __nv_bfloat16 actual;memcpy(&actual,&after[i][row*width[i]+col],2);
        maxerr=std::max(maxerr,std::abs(double(__bfloat162float(actual))-sum));
      }
    }
    if(maxerr>.003)return 5;
    printf("correctness=passed rows=%d shape=%d bitwise=true oracle_maxabs=%.12g\n",rows,s,maxerr);
    if(!sanitize&&rows>0&&rows<=64){
      cudaStream_t stream;ck(cudaStreamCreate(&stream));cudaGraph_t g[2];cudaGraphExec_t e[2];
      for(int t=0;t<2;t++){ck(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));for(int z=0;z<20;z++)run(t?v:0,stream);ck(cudaStreamEndCapture(stream,&g[t]));ck(cudaGraphInstantiate(&e[t],g[t],nullptr,nullptr,0));ck(cudaGraphLaunch(e[t],stream));}
      ck(cudaStreamSynchronize(stream));cudaEvent_t a,b;ck(cudaEventCreate(&a));ck(cudaEventCreate(&b));
      for(int trial=0;trial<5;trial++){float us[2];for(int o=0;o<2;o++){int t=(trial%2)?1-o:o;ck(cudaEventRecord(a,stream));ck(cudaGraphLaunch(e[t],stream));ck(cudaEventRecord(b,stream));ck(cudaEventSynchronize(b));float ms;ck(cudaEventElapsedTime(&ms,a,b));us[t]=ms*50;}
        printf("tokens=%d rows=%d trial=%d old_us=%.9g new_us=%.9g bitwise=true maxabs=0 oracle_maxabs=%.12g\n",rows,rows,trial,us[0],us[1],maxerr);}
      ck(cudaEventDestroy(a));ck(cudaEventDestroy(b));for(int t=0;t<2;t++){ck(cudaGraphExecDestroy(e[t]));ck(cudaGraphDestroy(g[t]));}ck(cudaStreamDestroy(stream));
    }
  }
  for(int i=0;i<6;i++){ck(cudaFree(w[i]));ck(cudaFree(y[i]));}ck(cudaFree(c));ck(cudaFree(x));ck(cudaDeviceReset());puts("released=true");
}
