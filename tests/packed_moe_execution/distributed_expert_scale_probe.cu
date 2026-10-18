#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
namespace old {
#include "baseline.cuh"
}
namespace next {
using old::lf_precision_e4;
using old::lf_precision_encode_e2;
#include "candidate.cuh"
}
static void ck(cudaError_t e) {if(e!=cudaSuccess){fprintf(stderr,"%s\n",cudaGetErrorString(e));exit(2);}}
template<class T> static T *alloc(size_t n){T *p;ck(cudaMalloc(&p,n*sizeof(T)));return p;}
static constexpr int R=old::R,H=old::H,I=old::I,S=old::S;
static_assert(R==next::R&&H==next::H&&I==next::I&&S==next::S);
static_assert(old::activation_bytes==next::activation_bytes&&old::stride==next::stride);
struct Frame {
  int *counts,*indices,*map,*error;
  __nv_bfloat16 *x,*activation;
  unsigned char *qx,*qa;
  float *weights,*weighted,*output;
  Frame(){
    counts=alloc<int>(5);indices=alloc<int>(R*S);map=alloc<int>(288);error=alloc<int>(1);
    x=alloc<__nv_bfloat16>(R*H);activation=alloc<__nv_bfloat16>(old::activation_bytes/2);
    qx=alloc<unsigned char>(next::physical_qb0);qa=alloc<unsigned char>(next::physical_qb1);
    weights=alloc<float>(R*S);weighted=alloc<float>(R*S*H);output=alloc<float>(R*H);
  }
  void close(){void *buffers[]={counts,indices,map,error,x,activation,qx,qa,weights,weighted,output};for(void *p:buffers)ck(cudaFree(p));}
};
// All launch dimensions come from each actual generated source. No geometry
// from the retained single-partition implementation is reused for the candidate.
#define CHAIN(NS,F,B,STREAM) \
  NS::next_group<<<NS::grid0,NS::block0,0,STREAM>>>(F.counts,F.indices,F.activation); \
  NS::next_quantize_0_global<<<NS::grid1,NS::block1,0,STREAM>>>(F.counts,F.x,F.qx,F.error); \
  NS::next_quantize_0_pack<<<NS::grid2,NS::block2,0,STREAM>>>(F.counts,F.x,F.qx,F.error); \
  NS::next_pack_rows_0<<<NS::grid3,NS::block3,0,STREAM>>>(F.counts,F.qx,F.activation); \
  NS::next_gate_up<<<NS::grid4,NS::block4,0,STREAM>>>(F.counts,F.qx,F.indices,F.map,B,F.activation); \
  NS::next_quantize_1_global<<<NS::grid5,NS::block5,0,STREAM>>>(F.counts,F.activation,F.qa,F.error); \
  NS::next_quantize_1_pack<<<NS::grid6,NS::block6,0,STREAM>>>(F.counts,F.activation,F.qa,F.error); \
  NS::next_pack_rows_1<<<NS::grid7,NS::block7,0,STREAM>>>(F.counts,F.qa,F.activation); \
  NS::next_down<<<NS::grid8,NS::block8,0,STREAM>>>(F.counts,F.qa,F.indices,F.weights,F.map,B,F.weighted,F.activation); \
  NS::next_combine<<<NS::grid9,NS::block9,0,STREAM>>>(F.counts,F.indices,F.weighted,F.output,F.weights)
static size_t differences(const char *name,const void *a,const void *b,size_t n){
  std::vector<unsigned char>x(n),y(n);ck(cudaMemcpy(x.data(),a,n,cudaMemcpyDeviceToHost));ck(cudaMemcpy(y.data(),b,n,cudaMemcpyDeviceToHost));
  size_t count=0,first=n;for(size_t i=0;i<n;i++)if(x[i]!=y[i]){if(!count)first=i;count++;}
  printf("frontier=%s bytes=%zu differences=%zu first=%zu\n",name,n,count,first);return count;
}
int main(int argc,char **argv){
  bool sanitize=argc>1&&strcmp(argv[1],"sanitize")==0;
  bool concurrent=argc>1&&strcmp(argv[1],"concurrent")==0;
  constexpr int banks=8;
  std::vector<unsigned char> bank(old::stride*banks);
  const int widths[3]={H,H,I},outs[3]={I,I,H};
  const size_t scales[3]={old::ws0,old::ws1,old::ws2},globals[3]={old::wg0,old::wg1,old::wg2};
  for(int e=0;e<banks;e++)for(int s=0;s<3;s++){
    auto *p=bank.data()+e*old::stride+old::offsets[s];
    for(int r=0;r<outs[s];r++){
      for(int k=0;k<widths[s];k+=2)p[(r*widths[s]+k)/2]=((r*3+k*5+e+s)&15)|(((r*3+(k+1)*5+e+s)&15)<<4);
      for(int k=0;k<widths[s]/16;k++)p[scales[s]+r*(widths[s]/16)+k]=32+(r+k+e+s)%5;
    }
    float global=.03125f;memcpy(p+globals[s],&global,4);
  }
  auto *raw=alloc<unsigned char>(bank.size()),*packed=alloc<unsigned char>(bank.size()),*scratch=alloc<unsigned char>(old::stride);
  // Match checkpoint upload: only defined planes/scalars are initialized,
  // never allocator alignment padding. Initcheck must accept preparation.
  for(int e=0;e<banks;e++)for(int s=0;s<3;s++){
    size_t offset=e*old::stride+old::offsets[s];
    ck(cudaMemcpy(raw+offset,bank.data()+offset,globals[s]+4,cudaMemcpyHostToDevice));
  }
  for(int e=0;e<banks;e++){old::next_prepare_bank<<<256,256>>>(raw+e*old::stride,scratch);old::next_prepare_bank_copy<<<256,256>>>(scratch,packed+e*old::stride);}
  ck(cudaDeviceSynchronize());ck(cudaFree(raw));ck(cudaFree(scratch));
  Frame a,b;cudaStream_t stream,auxiliary;ck(cudaStreamCreate(&stream));ck(cudaStreamCreate(&auxiliary));
  cudaGraph_t graphs[2];cudaGraphExec_t execs[2];
  for(int i=0;i<2;i++){
    ck(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));
    if(i){CHAIN(next,b,packed,stream);}else{CHAIN(old,a,packed,stream);}
    ck(cudaStreamEndCapture(stream,&graphs[i]));ck(cudaGraphInstantiate(&execs[i],graphs[i],nullptr,nullptr,0));
  }
  std::vector<__nv_bfloat16>x(R*H);std::vector<int>ids(R*S),map(288);std::vector<float>weights(R*S);
  // Vary occupancy, live-count transitions and exact BF16 magnitudes; preserve
  // scratch between steps as a resident graph does. This is not a model oracle.
  int live_counts[]={22,1,1,36,1,3,64,1,23,24,1};
  for(int step=0;step<(sanitize?2:44);step++){
    int live=live_counts[step%11],c[5]={0,0,0,live,0};
    for(int i=0;i<R*H;i++)x[i]=__float2bfloat16_rn(float(int((i*7919+step*53)%65521)-32760)/(step%3?1024.f:32.f));
    for(int i=0;i<R*S;i++){ids[i]=(i*7+step*13)%288;weights[i]=float(1+i%13)/64.f;}
    for(int i=0;i<288;i++)map[i]=step%2&&i%3==0?-1:i%banks;
    for(Frame *f:{&a,&b}){
      ck(cudaMemcpy(f->counts,c,sizeof(c),cudaMemcpyHostToDevice));ck(cudaMemcpy(f->x,x.data(),x.size()*2,cudaMemcpyHostToDevice));
      ck(cudaMemcpy(f->indices,ids.data(),ids.size()*4,cudaMemcpyHostToDevice));ck(cudaMemcpy(f->map,map.data(),map.size()*4,cudaMemcpyHostToDevice));
      ck(cudaMemcpy(f->weights,weights.data(),weights.size()*4,cudaMemcpyHostToDevice));ck(cudaMemset(f->error,0,4));
      // Poison output storage identically; only active output is inspected.
      ck(cudaMemset(f->qx,0xa5,next::physical_qb0));ck(cudaMemset(f->qa,0xa5,next::physical_qb1));
    }
    ck(cudaGraphLaunch(execs[step%2],stream));ck(cudaGraphLaunch(execs[1-step%2],concurrent?auxiliary:stream));
    ck(cudaStreamSynchronize(stream));ck(cudaStreamSynchronize(auxiliary));
    printf("step=%d live=%d concurrent=%s\n",step,live,concurrent?"true":"false");
    size_t errors=0;
    errors+=differences("quantized_input",a.qx,b.qx,old::qb0);
    errors+=differences("gate_up",a.activation,b.activation,(size_t)live*S*I*2);
    errors+=differences("quantized_activation",a.qa,b.qa,old::qb1);
    errors+=differences("down",a.weighted,b.weighted,(size_t)live*S*H*4);
    errors+=differences("combine",a.output,b.output,(size_t)live*H*4);
    int e0,e1;ck(cudaMemcpy(&e0,a.error,4,cudaMemcpyDeviceToHost));ck(cudaMemcpy(&e1,b.error,4,cudaMemcpyDeviceToHost));
    if(errors||e0||e1){fprintf(stderr,"first divergent step=%d error=%d/%d\n",step,e0,e1);return 4;}
  }
  for(int i=0;i<2;i++){ck(cudaGraphExecDestroy(execs[i]));ck(cudaGraphDestroy(graphs[i]));}
  ck(cudaStreamDestroy(stream));ck(cudaStreamDestroy(auxiliary));a.close();b.close();ck(cudaFree(packed));ck(cudaDeviceReset());puts("released=true");
}
