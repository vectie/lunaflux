#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <nccl.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "experts.cuh"

static void ck(cudaError_t e) { if(e!=cudaSuccess) {fprintf(stderr,"CUDA %s\n",cudaGetErrorString(e));exit(1);} }
static void nk(ncclResult_t e) { if(e!=ncclSuccess) {fprintf(stderr,"NCCL %s\n",ncclGetErrorString(e));exit(1);} }
template<class T> static T* alloc(size_t n) { T*p; ck(cudaMalloc(&p,n*sizeof(T))); return p; }

// The source coordinates are global; every rank keeps the same expert IDs.
// Payloads/scales remain packed. This fixture has no model checkpoint or hash.
static std::vector<unsigned char> bank(bool partial,int rank) {
  const size_t stride=partial?shard_stride:whole_stride;
  const size_t* offsets=partial?shard_offsets:whole_offsets;
  std::vector<unsigned char> out(stride*2,0);
  for(int e=0;e<2;++e) for(int stage=0;stage<3;++stage) {
    int n=partial&&stage<2?32:64,k=partial&&stage==2?32:64;
    auto*p=out.data()+stride*e+offsets[stage];
    for(int r=0;r<n;++r) for(int c=0;c<k;c+=2) {
      int rr=r+(partial&&stage<2?rank*32:0),cc=c+(partial&&stage==2?rank*32:0);
      unsigned lo=2+((rr+cc+e+stage)%3),hi=2+((rr+cc+1+e+stage)%3);
      p[r*k/2+c/2]=(unsigned char)(lo|(hi<<4));
    }
    memset(p+n*k/2,0x38,n*(k/16));
    float scale=.03125f; memcpy(p+n*k/2+n*(k/16),&scale,4);
  }
  return out;
}

int main(int argc,char**argv) {
  if(argc!=3)return 2;
  int rank=atoi(argv[1]); if(rank<0||rank>1)return 2;
  // The bootstrap listener belongs to rank zero and stays alive through init.
  // A separate GetUniqueId process must not exit before the ranks connect.
  ncclUniqueId id;
  if(rank==0) {
    nk(ncclGetUniqueId(&id));FILE*f=fopen(argv[2],"wbx");if(!f)return 3;
    if(fwrite(&id,1,sizeof(id),f)!=sizeof(id))return 3;fclose(f);
  } else {
    FILE*f=fopen(argv[2],"rb");if(!f)return 3;
    if(fread(&id,1,sizeof(id),f)!=sizeof(id))return 3;fclose(f);
  }
  ck(cudaSetDevice(0));cudaStream_t stream;ck(cudaStreamCreate(&stream));
  ncclComm_t comm;nk(ncclCommInitRank(&comm,2,id,rank));
  auto full=bank(false,0),local=bank(true,rank);
  auto wf=alloc<unsigned char>(full.size()),wl=alloc<unsigned char>(local.size());
  ck(cudaMemcpy(wf,full.data(),full.size(),cudaMemcpyHostToDevice));
  ck(cudaMemcpy(wl,local.data(),local.size(),cudaMemcpyHostToDevice));
  auto counts=alloc<int>(5),indices=alloc<int>(6),map=alloc<int>(2);
  auto weights=alloc<float>(6);auto x=alloc<__nv_bfloat16>(3*64);
  int hc[]={0,1,1,3,0},hi[]={0,1,1,0,0,1},hm[]={0,1};
  float hw[]={.25f,.75f,.625f,.375f,.5f,.5f};
  std::vector<__nv_bfloat16> hx(3*64);for(int i=0;i<3*64;++i)hx[i]=__float2bfloat16_rn((i%7-3)*.03125f);
  ck(cudaMemcpy(indices,hi,sizeof(hi),cudaMemcpyHostToDevice));ck(cudaMemcpy(map,hm,sizeof(hm),cudaMemcpyHostToDevice));
  ck(cudaMemcpy(weights,hw,sizeof(hw),cudaMemcpyHostToDevice));ck(cudaMemcpy(x,hx.data(),hx.size()*2,cudaMemcpyHostToDevice));
  auto af=alloc<unsigned char>(whole_activation),al=alloc<unsigned char>(shard_activation);
  auto rounded=alloc<__nv_bfloat16>(6*64);auto partial=alloc<float>(6*64);
  auto yf=alloc<float>(3*64),yl=alloc<float>(3*64);
  for(int rows:{1,3}) {
    hc[3]=rows;ck(cudaMemcpy(counts,hc,sizeof(hc),cudaMemcpyHostToDevice));
    whole_group<<<whole_grid0,whole_block0,0,stream>>>(counts,indices,(__nv_bfloat16*)af);
    whole_gate_up<<<whole_grid1,whole_block1,0,stream>>>(counts,x,indices,map,wf,(__nv_bfloat16*)af);
    whole_down<<<whole_grid2,whole_block2,0,stream>>>(counts,(__nv_bfloat16*)af,indices,weights,map,wf,rounded);
    whole_combine<<<whole_grid3,whole_block3,0,stream>>>(counts,indices,rounded,yf);
    shard_group<<<shard_grid0,shard_block0,0,stream>>>(counts,indices,(__nv_bfloat16*)al);
    shard_gate_up<<<shard_grid1,shard_block1,0,stream>>>(counts,x,indices,map,wl,(__nv_bfloat16*)al);
    shard_down<<<shard_grid2,shard_block2,0,stream>>>(counts,(__nv_bfloat16*)al,indices,weights,map,wl,partial);
    nk(ncclAllReduce(partial,partial,rows*2*64,ncclFloat32,ncclSum,comm,stream));
    shard_combine<<<shard_grid3,shard_block3,0,stream>>>(counts,indices,partial,yl,weights);
    ck(cudaGetLastError());ck(cudaStreamSynchronize(stream));
    std::vector<float> ref(rows*64),actual(rows*64);
    ck(cudaMemcpy(ref.data(),yf,ref.size()*4,cudaMemcpyDeviceToHost));ck(cudaMemcpy(actual.data(),yl,actual.size()*4,cudaMemcpyDeviceToHost));
    size_t exact=0;for(size_t i=0;i<ref.size();++i)if(!memcmp(&ref[i],&actual[i],4))++exact;
    printf("rank=%d rows=%d exact=%zu/%zu reduce_before_bf16=true\n",rank,rows,exact,ref.size());fflush(stdout);
    if(exact!=ref.size())return 4;
  }
  nk(ncclCommDestroy(comm));
  for(void*p:{(void*)wf,(void*)wl,(void*)counts,(void*)indices,(void*)map,(void*)weights,(void*)x,(void*)af,(void*)al,(void*)rounded,(void*)partial,(void*)yf,(void*)yl})ck(cudaFree(p));
  ck(cudaStreamDestroy(stream));ck(cudaDeviceReset());puts("released=true");
}
