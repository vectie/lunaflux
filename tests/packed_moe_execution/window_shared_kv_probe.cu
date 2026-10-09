// Bounded shared-KV causal window oracle. Component execution, not model speed.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "window_kernels.cuh"

static void check(cudaError_t error) {
  if (error != cudaSuccess) { std::fprintf(stderr,"%s\n",cudaGetErrorString(error)); std::exit(2); }
}
template<class T> static T *allocate(size_t count) {
  T *result=nullptr; check(cudaMallocManaged(&result,count*sizeof(T)));
  std::memset(result,0,count*sizeof(T)); return result;
}
static float bfloat(float value) { return __bfloat162float(__float2bfloat16_rn(value)); }
static unsigned short word(float value) {
  auto stored=__float2bfloat16_rn(value); unsigned short result;
  std::memcpy(&result,&stored,2); return result;
}
int main() {
  constexpr int R=256,H=2,D=512,W=128;
  auto counts=allocate<int>(5),reset=allocate<int>(1),positions=allocate<int>(R);
  auto history=allocate<int>(2),append=allocate<int>(3);
  auto query=allocate<__nv_bfloat16>(R*H*D),kv=allocate<__nv_bfloat16>(R*D);
  auto ring=allocate<unsigned short>(W*D);
  auto output=allocate<__nv_bfloat16>(R*H*D);
  auto sink=allocate<float>(H); sink[0]=0.3f; sink[1]=-1.5f;
  std::vector<float> reference;
  for (int round=0;round<2;++round) {
    reference.clear(); history[0]=0; history[1]=0;
    std::fill(ring,ring+W*D,word(19.f));
    for (int live : {256,13,1,127,256,0,3}) {
      const bool restart=live==3;
      if (restart) reference.clear();
      const int base=int(reference.size()/D);
      counts[2]=1; counts[3]=live; reset[0]=restart?1:0;
      for (int row=0;row<live;++row) {
        positions[row]=base+row;
        for (int component=0;component<D;++component) {
          float value=bfloat(float(((base+row)*11+component*7)%43-21)*0.03125f);
          kv[row*D+component]=__float2bfloat16_rn(value); reference.push_back(value);
          for (int head=0;head<H;++head)
            query[(row*H+head)*D+component]=__float2bfloat16_rn(float(((base+row)*3+head*13+component*5)%29-14)*0.0625f);
        }
      }
      std::fill(output,output+R*H*D,__float2bfloat16_rn(19.f));
      window_reserve<<<1,1>>>(counts,reset,positions,history,append);
      window_attention<<<dim3(R,H),128>>>(append,query,kv,(const __nv_bfloat16*)ring,sink,output);
      window_copy<<<W,128>>>(append,(const unsigned short*)kv,ring);
      window_publish<<<1,1>>>(append,history);
      check(cudaGetLastError()); check(cudaDeviceSynchronize());
      if (append[0]!=base || append[1]!=live || append[2]!=0 || history[0]!=base+live || history[1]) return 3;
      float maximum=0.f;
      for (int row=0;row<live;++row) for (int head=0;head<H;++head) {
        const int position=base+row,first=std::max(0,position-W+1),length=position-first+1;
        std::vector<float> scores(length); float largest=sink[head];
        for (int slot=0;slot<length;++slot) {
          float dot=0.f;
          for (int component=0;component<D;++component)
            dot=dot+__bfloat162float(query[(row*H+head)*D+component])*reference[(first+slot)*D+component];
          scores[slot]=dot*(1.f/std::sqrt(float(D))); largest=std::max(largest,scores[slot]);
        }
        float denominator=std::exp(sink[head]-largest);
        for (float &score:scores) { score=std::exp(score-largest); denominator=denominator+score; }
        for (int component=0;component<D;++component) {
          float sum=0.f; for (int slot=0;slot<length;++slot) sum=sum+scores[slot]*reference[(first+slot)*D+component];
          float expected=bfloat(sum/denominator),observed=__bfloat162float(output[(row*H+head)*D+component]);
          maximum=std::max(maximum,std::fabs(expected-observed));
        }
      }
      if (maximum>0.0078125f) { std::fprintf(stderr,"maxabs=%.9g\n",maximum); return 4; }
      for (int cell=live*H*D;cell<R*H*D;++cell) if (__bfloat162float(output[cell])!=19.f) return 5;
      for (int absolute=std::max(0,base+live-W);absolute<base+live;++absolute)
        for (int component=0;component<D;++component)
          if (ring[(absolute%W)*D+component]!=word(reference[absolute*D+component])) return 6;
      std::printf("round=%d base=%d live=%d maxabs=%.9g ring=exact inactive=untouched\n",round,base,live,maximum);
    }
    // Reject a gap; poison persists until explicit device reset.
    counts[3]=1; reset[0]=0; positions[0]=history[0]+1;
    std::vector<unsigned short> before(ring,ring+W*D);
    window_reserve<<<1,1>>>(counts,reset,positions,history,append);
    window_copy<<<W,128>>>(append,(const unsigned short*)kv,ring);
    window_publish<<<1,1>>>(append,history);
    check(cudaDeviceSynchronize());
    if (!append[2] || !history[1] || std::memcmp(before.data(),ring,W*D*2)) return 7;
    positions[0]=history[0]; window_reserve<<<1,1>>>(counts,reset,positions,history,append);
    check(cudaDeviceSynchronize()); if (!append[2]) return 8;
    reset[0]=1; positions[0]=0;
    window_reserve<<<1,1>>>(counts,reset,positions,history,append);
    window_copy<<<W,128>>>(append,(const unsigned short*)kv,ring);
    window_publish<<<1,1>>>(append,history);
    check(cudaDeviceSynchronize()); if (append[2] || history[1] || history[0]!=1) return 9;
    reset[0]=0; history[0]=1048576; counts[3]=1; positions[0]=1048576;
    window_reserve<<<1,1>>>(counts,reset,positions,history,append);
    check(cudaDeviceSynchronize()); if (!append[2] || !history[1]) return 10;
  }
  // Tiny component replay timing only, not whole-model token throughput.
  counts[2]=1; counts[3]=R; reset[0]=1;
  for (int row=0;row<R;++row) positions[row]=row;
  cudaEvent_t begin,end; check(cudaEventCreate(&begin)); check(cudaEventCreate(&end));
  check(cudaEventRecord(begin));
  for (int repeat=0;repeat<10;++repeat) {
    window_reserve<<<1,1>>>(counts,reset,positions,history,append);
    window_attention<<<dim3(R,H),128>>>(append,query,kv,(const __nv_bfloat16*)ring,sink,output);
    window_copy<<<W,128>>>(append,(const unsigned short*)kv,ring);
    window_publish<<<1,1>>>(append,history);
  }
  check(cudaEventRecord(end)); check(cudaEventSynchronize(end));
  float millis=0.f; check(cudaEventElapsedTime(&millis,begin,end));
  if (append[2] || history[0]!=R || history[1]) return 11;
  std::printf("component replay rows=%d heads=%d width=%d window=%d ms=%.6f\n",R,H,D,W,millis/10.f);
  check(cudaEventDestroy(end)); check(cudaEventDestroy(begin));
  check(cudaFree(sink)); check(cudaFree(output)); check(cudaFree(ring));
  check(cudaFree(kv)); check(cudaFree(query)); check(cudaFree(append));
  check(cudaFree(history)); check(cudaFree(positions)); check(cudaFree(reset)); check(cudaFree(counts));
  std::puts("shared window correctness passed; all allocations released");
  return 0;
}
