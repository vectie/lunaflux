#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <nccl.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "experts.cuh"

// Offline test of the ACTUAL exported publication kernels and BF16 NCCL.
// Synthetic rank-local operands test the declared new arithmetic, not parity
// with the old F32-per-expert program or model quality.
static void ck(cudaError_t e) { if(e!=cudaSuccess) {fprintf(stderr,"CUDA %s\n",cudaGetErrorString(e));exit(1);} }
static void nk(ncclResult_t e) { if(e!=ncclSuccess) {fprintf(stderr,"NCCL %s\n",ncclGetErrorString(e));exit(1);} }
template<class T> static T* alloc(size_t n) { T*p; ck(cudaMalloc(&p,n*sizeof(T))); return p; }
static float bf(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }
static float add(float a,float b) { volatile float x=a+b; return x; }
static float mul(float a,float b) { volatile float x=a*b; return x; }
static float routed(int rank,int row,int slot,int col) {
  return ((col*7+slot*11+row*3+rank*13)%127-63)*0.00317f;
}
static float shared(int rank,int row,int col) {
  return ((col*3+row+rank*17)%71-35)*0.00239f;
}
static float route_weight(int slot) { return (slot+1)*0.027f; }
static __nv_bfloat16 reference(int row,int col,int selected_rank=-1) {
  float local[2];
  for(int rank=0;rank<2;++rank) {
    float s=0;
    // IDs are a permutation: 7,0,6,1,5,2,4,3. Preserve ascending expert sum.
    for(int slot:{1,3,5,7,6,4,2,0})
      s=add(s,bf(mul(bf(routed(rank,row,slot,col)),route_weight(slot))));
    local[rank]=bf(add(bf(s),bf(shared(rank,row,col))));
  }
  return __float2bfloat16_rn(selected_rank<0?add(local[0],local[1]):local[selected_rank]);
}

int main(int argc,char**argv) {
  bool local_only=argc==4&&!strcmp(argv[3],"--local-only");
  if(argc!=3&&!local_only)return 2;
  int rank=atoi(argv[1]);if(rank<0||rank>1)return 2;
  ncclUniqueId id;
  if(rank==0) {
    nk(ncclGetUniqueId(&id));FILE*f=fopen(argv[2],"wbx");if(!f)return 3;
    if(fwrite(&id,1,sizeof(id),f)!=sizeof(id))return 3;fclose(f);
  } else {
    FILE*f=fopen(argv[2],"rb");if(!f)return 3;
    if(fread(&id,1,sizeof(id),f)!=sizeof(id))return 3;fclose(f);
  }
  ck(cudaSetDevice(0));cudaStream_t main,branch;
  ck(cudaStreamCreateWithFlags(&main,cudaStreamNonBlocking));
  ck(cudaStreamCreateWithFlags(&branch,cudaStreamNonBlocking));
  cudaEvent_t joined;ck(cudaEventCreateWithFlags(&joined,cudaEventDisableTiming));
  ncclComm_t comm=nullptr;if(!local_only)nk(ncclCommInitRank(&comm,2,id,rank));
  constexpr int rows=64,h=4096,k=8;
  auto counts=alloc<int>(5),indices=alloc<int>(rows*k),shared_ids=alloc<int>(rows);
  auto weights=alloc<float>(rows*k),unit=alloc<float>(rows);
  auto partial=alloc<float>(rows*k*h),sp=alloc<float>(rows*h);
  auto ro=alloc<float>(rows*h),so=alloc<float>(rows*h);
  auto output=alloc<__nv_bfloat16>(rows*h);
  std::vector<int> hi(rows*k),hs(rows,0);int ids[]={7,0,6,1,5,2,4,3};
  std::vector<float> hw(rows*k),hu(rows,1),hp(rows*k*h),hsp(rows*h);
  for(int r=0;r<rows;++r) {
    for(int j=0;j<k;++j) {
      hi[r*k+j]=ids[j];hw[r*k+j]=route_weight(j);
      for(int c=0;c<h;++c) hp[(r*k+j)*h+c]=routed(rank,r,j,c);
    }
    for(int c=0;c<h;++c)hsp[r*h+c]=shared(rank,r,c);
  }
  ck(cudaMemcpy(indices,hi.data(),hi.size()*4,cudaMemcpyHostToDevice));
  ck(cudaMemcpy(shared_ids,hs.data(),hs.size()*4,cudaMemcpyHostToDevice));
  ck(cudaMemcpy(weights,hw.data(),hw.size()*4,cudaMemcpyHostToDevice));
  ck(cudaMemcpy(unit,hu.data(),hu.size()*4,cudaMemcpyHostToDevice));
  ck(cudaMemcpy(partial,hp.data(),hp.size()*4,cudaMemcpyHostToDevice));
  ck(cudaMemcpy(sp,hsp.data(),hsp.size()*4,cudaMemcpyHostToDevice));
  for(int live:{1,3,64,2}) {
    int hc[]={0,1,1,live,0};ck(cudaMemcpy(counts,hc,sizeof(hc),cudaMemcpyHostToDevice));
    ck(cudaMemset(output,0x55,rows*h*2));
    glm_routed_3_combine<<<rows*h/256,256,0,main>>>(counts,indices,partial,ro,weights);
    glm_shared_3_combine<<<rows*h/256,256,0,branch>>>(counts,shared_ids,sp,so,unit);
    ck(cudaEventRecord(joined,branch));ck(cudaStreamWaitEvent(main,joined,0));
    glm_moe_combine_3<<<rows*h/256,256,0,main>>>(counts,ro,so,output);
    if(!local_only)nk(ncclAllReduce(output,output,live*h,ncclBfloat16,ncclSum,comm,main));
    ck(cudaGetLastError());ck(cudaStreamSynchronize(main));
    std::vector<__nv_bfloat16> actual(rows*h);ck(cudaMemcpy(actual.data(),output,rows*h*2,cudaMemcpyDeviceToHost));
    size_t exact=0;for(int r=0;r<live;++r)for(int c=0;c<h;++c){
      auto expected=reference(r,c,local_only?rank:-1);if(!memcmp(&expected,&actual[r*h+c],2))++exact;
    }
    for(int i=live*h;i<rows*h;++i){unsigned short bits;memcpy(&bits,&actual[i],2);if(bits!=0x5555)return 5;}
    printf("rank=%d live=%d exact=%zu/%d inactive_unchanged=true bytes=%d collective_executed=%s\n",rank,live,exact,live*h,live*h*2,local_only?"false":"true");fflush(stdout);
    if(exact!=size_t(live*h))return 4;
  }
  if(comm)nk(ncclCommDestroy(comm));
  for(void*p:{(void*)counts,(void*)indices,(void*)shared_ids,(void*)weights,(void*)unit,(void*)partial,(void*)sp,(void*)ro,(void*)so,(void*)output})ck(cudaFree(p));
  ck(cudaEventDestroy(joined));ck(cudaStreamDestroy(branch));ck(cudaStreamDestroy(main));ck(cudaDeviceReset());puts("released=true");
}
