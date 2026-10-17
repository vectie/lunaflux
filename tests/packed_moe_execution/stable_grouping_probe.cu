#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <vector>
#include <cstdlib>
#include "candidate.cuh"
static void ck(cudaError_t e) {
  if(e!=cudaSuccess) {fprintf(stderr,"%s\n",cudaGetErrorString(e));exit(2);}
}
int main() {
  int *counts,*indices; __nv_bfloat16 *activation;
  ck(cudaMalloc(&counts,5*sizeof(int)));ck(cudaMalloc(&indices,R*S*sizeof(int)));
  ck(cudaMalloc(&activation,activation_bytes));
  constexpr int jobs=R*S, meta_count=1+jobs*17;
  size_t metadata=activation_bytes-meta_count*sizeof(int);
  int cases=0;
  for(int rows: {0,1,3,16,17,R,R+1}) for(int pattern=0;pattern<5;pattern++) {
    std::vector<int> ids(jobs),expected(meta_count,-1),actual(meta_count);
    for(int j=0;j<jobs;j++) ids[j]=pattern==0?4:pattern==1?j:pattern==2?(j*7)%23:pattern==3?j%S:(j*71+5)%288;
    int live=rows>0&&rows<=R?rows*S:0,groups=0;
    std::vector<int> seen;
    for(int j=0;j<live;j++) {
      bool exists=false;for(int e:seen) exists|=e==ids[j];if(exists)continue;
      seen.push_back(ids[j]);int rank=0;
      for(int k=j;k<live;k++) if(ids[k]==ids[j]) {
        int group=groups+rank/16;expected[1+group]=ids[j];
        expected[1+jobs+group*16+rank%16]=k;rank++;
      }
      groups+=(rank+15)/16;
    }
    expected[0]=groups;int c[5]={0,0,0,rows,0};
    ck(cudaMemcpy(counts,c,sizeof(c),cudaMemcpyHostToDevice));
    ck(cudaMemcpy(indices,ids.data(),ids.size()*4,cudaMemcpyHostToDevice));
    next_group<<<1,128>>>(counts,indices,activation);ck(cudaGetLastError());
    ck(cudaMemcpy(actual.data(),reinterpret_cast<char*>(activation)+metadata,meta_count*4,cudaMemcpyDeviceToHost));
    if(actual!=expected) {fprintf(stderr,"metadata mismatch rows=%d pattern=%d\n",rows,pattern);return 3;}
    cases++;
  }
  ck(cudaFree(counts));ck(cudaFree(indices));ck(cudaFree(activation));ck(cudaDeviceReset());
  printf("stable_assignment_cases=%d exact_integer_oracle=true released=true\n",cases);
}
