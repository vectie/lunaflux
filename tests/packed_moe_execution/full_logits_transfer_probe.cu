#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include "source.cuh"
#define CK(x) do { auto e=(x); if(e!=cudaSuccess) { std::fprintf(stderr,"%s\n",cudaGetErrorString(e)); std::exit(2); } } while(0)
int main() {
  int *counts,*n,*selected;
  float *local,*all,*out;
  CK(cudaMalloc(&counts,20)); CK(cudaMalloc(&n,4)); CK(cudaMalloc(&selected,12));
  CK(cudaMalloc(&local,72)); CK(cudaMalloc(&all,216)); CK(cudaMalloc(&out,204));
  for(int live: {0,1,3,2,1}) {
    int c[5]={0,0,2,4,0},s[3]={0,2,3};
    CK(cudaMemcpy(counts,c,20,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(n,&live,4,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(selected,s,12,cudaMemcpyHostToDevice));
    CK(cudaMemset(all,0xff,216)); CK(cudaMemset(out,0,204));
    for(int rank=0;rank<3;++rank) {
      int first=17*rank/3,end=17*(rank+1)/3,columns=end-first;
      float x[18]; for(int i=0;i<18;++i)x[i]=-999.0f;
      for(int row=0;row<live;++row)for(int word=0;word<columns;++word)
        x[row*columns+word]=(row*17+first+word-20)*0.125f;
      CK(cudaMemcpy(local,x,72,cudaMemcpyHostToDevice));
      pack(rank,counts,n,selected,local,all+rank*18);
      CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    }
    float transport[54]; CK(cudaMemcpy(transport,all,216,cudaMemcpyDeviceToHost));
    for(int rank=0;rank<3;++rank)for(int row=0;row<3;++row)for(int word=0;word<6;++word) {
      const int first=17*rank/3,columns=17*(rank+1)/3-first;
      const float expected=row<live&&word<columns?(row*17+first+word-20)*0.125f:0.0f;
      if(transport[rank*18+row*6+word]!=expected) return 3;
    }
    join(n,all,out); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    float actual[51]; CK(cudaMemcpy(actual,out,204,cudaMemcpyDeviceToHost));
    for(int row=0;row<live;++row)for(int word=0;word<17;++word)
      if(actual[row*17+word]!=(row*17+word-20)*0.125f) return 4;
    std::printf("live=%d transport_exact=true logits_exact=true\n",live);
  }
  CK(cudaFree(out)); CK(cudaFree(all)); CK(cudaFree(local));
  CK(cudaFree(selected)); CK(cudaFree(n)); CK(cudaFree(counts));
  CK(cudaDeviceReset()); std::puts("released=true");
}
