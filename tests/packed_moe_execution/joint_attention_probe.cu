// Independent bounded joint-softmax oracle. This is not full-model validation.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "joint_attention_kernels.cuh"

static void check(cudaError_t error) {
  if (error != cudaSuccess) { std::fprintf(stderr,"%s\n",cudaGetErrorString(error)); std::exit(2); }
}
template<class T> static T *allocate(size_t count) {
  T *result=nullptr; check(cudaMallocManaged(&result,count*sizeof(T)));
  std::memset(result,0,count*sizeof(T)); return result;
}
static float bf(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }
static unsigned short word(float x) {
  const auto value=__float2bfloat16_rn(x); unsigned short result;
  std::memcpy(&result,&value,2); return result;
}

template<int Mode, int R, int W, int Max, int Ratio, int K> static void run() {
  constexpr int H=2,D=128,C=Max/Ratio,PhysicalC=C?C:1,Slots=K?K:1;
  auto counts=allocate<int>(5),reset=allocate<int>(1),positions=allocate<int>(R);
  auto cache_count=allocate<int>(5),cache_append=allocate<int>(3);
  auto selected_count=allocate<int>(R),ids=allocate<int>(R*Slots),offsets=allocate<int>(R);
  auto history=allocate<int>(2),append=allocate<int>(3);
  auto queries=allocate<__nv_bfloat16>(R*H*D),current=allocate<__nv_bfloat16>(R*D);
  auto compressed=allocate<__nv_bfloat16>(PhysicalC*D),output=allocate<__nv_bfloat16>(R*H*D);
  auto ring=allocate<unsigned short>(W*D); auto sink=allocate<float>(H);
  sink[0]=0.35f; sink[1]=-1.2f;
  for (int i=0;i<PhysicalC*D;++i) compressed[i]=__float2bfloat16_rn(float((i*7)%31-15)*0.03125f);
  auto execute=[&]() {
    if constexpr(Mode==4) {
      joint4_reserve<<<1,1>>>(counts,reset,positions,cache_count,cache_append,selected_count,ids,offsets,history,append);
      joint4_attention<<<dim3(R,H),128>>>(append,selected_count,ids,offsets,queries,current,(const __nv_bfloat16*)ring,compressed,sink,output);
      joint4_copy<<<W,128>>>(append,(const unsigned short*)current,ring);
      joint4_publish<<<1,1>>>(append,history);
    } else if constexpr(Mode==128) {
      joint128_reserve<<<1,1>>>(counts,reset,positions,cache_count,cache_append,selected_count,ids,offsets,history,append);
      joint128_attention<<<dim3(R,H),128>>>(append,selected_count,ids,offsets,queries,current,(const __nv_bfloat16*)ring,compressed,sink,output);
      joint128_copy<<<W,128>>>(append,(const unsigned short*)current,ring);
      joint128_publish<<<1,1>>>(append,history);
    } else {
      joint0_reserve<<<1,1>>>(counts,reset,positions,cache_count,cache_append,selected_count,ids,offsets,history,append);
      joint0_attention<<<dim3(R,H),128>>>(append,selected_count,ids,offsets,queries,current,(const __nv_bfloat16*)ring,compressed,sink,output);
      joint0_copy<<<W,128>>>(append,(const unsigned short*)current,ring);
      joint0_publish<<<1,1>>>(append,history);
    }
    check(cudaGetLastError()); check(cudaDeviceSynchronize());
  };
  int steps=0;
  for(int replay=0;replay<2;++replay) {
    std::vector<float> raw; history[0]=history[1]=0;
    std::vector<int> chunks;
    if constexpr(Mode==128) { chunks.assign(18,8); chunks.push_back(0); chunks.push_back(1); }
    else if constexpr(Mode==4) { chunks={3,8,1,8,0,3}; }
    else { chunks={1,2,0}; }
    for(int live:chunks) {
      const int base=int(raw.size()/D);
      counts[2]=1; counts[3]=live; reset[0]=base==0;
      cache_count[2]=1; cache_count[3]=(base+live)/Ratio; cache_append[2]=0;
      for(int row=0;row<R;++row) {
        offsets[row]=9;
        const int visible=(base+row+1)/Ratio;
        selected_count[row]=std::min(K,visible);
        for(int slot=0;slot<Slots;++slot) ids[row*Slots+slot]=slot<selected_count[row]?9+visible-1-slot:-1;
      }
      for(int row=0;row<live;++row) {
        positions[row]=base+row;
        for(int col=0;col<D;++col) {
          const float value=bf(float(((base+row)*11+col*7)%43-21)*0.03125f);
          current[row*D+col]=__float2bfloat16_rn(value); raw.push_back(value);
          for(int head=0;head<H;++head) queries[(row*H+head)*D+col]=__float2bfloat16_rn(float(((base+row)*3+head*13+col*5)%29-14)*0.0625f);
        }
      }
      std::fill(output,output+R*H*D,__float2bfloat16_rn(19.f)); execute();
      if(append[0]!=base || append[1]!=live || append[2] || history[0]!=base+live || history[1]) std::exit(3);
      float largest_error=0.f;
      for(int row=0;row<live;++row) for(int head=0;head<H;++head) {
        const int position=base+row,first=std::max(0,position-W+1),n=position-first+1;
        const int visible=(position+1)/Ratio,nc=Mode==128?visible:selected_count[row];
        auto value=[&](int ordinal,int col) {
          if(ordinal<n) return raw[(first+ordinal)*D+col];
          const int slot=ordinal-n,index=Mode==128?slot:ids[row*Slots+slot]-offsets[row];
          return __bfloat162float(compressed[index*D+col]);
        };
        std::vector<float> scores(n+nc); float maximum=sink[head];
        for(int slot=0;slot<n+nc;++slot) {
          float dot=0.f;
          for(int col=0;col<D;++col) dot=dot+__bfloat162float(queries[(row*H+head)*D+col])*value(slot,col);
          scores[slot]=dot*float(1.0/std::sqrt(double(D))); maximum=std::max(maximum,scores[slot]);
        }
        float denominator=std::exp(sink[head]-maximum);
        for(float &score:scores) { score=std::exp(score-maximum); denominator=denominator+score; }
        for(int col=0;col<D;++col) {
          float sum=0.f; for(int slot=0;slot<n+nc;++slot) sum=sum+scores[slot]*value(slot,col);
          const float expected=bf(sum/denominator),observed=__bfloat162float(output[(row*H+head)*D+col]);
          largest_error=std::max(largest_error,std::fabs(expected-observed));
        }
      }
      if(largest_error>0.00390625f) { std::fprintf(stderr,"mode=%d base=%d maxabs=%.9g\n",Mode,base,largest_error); std::exit(4); }
      for(int i=live*H*D;i<R*H*D;++i) if(word(__bfloat162float(output[i]))!=0) std::exit(5);
      for(int pos=std::max(0,base+live-W);pos<base+live;++pos) for(int col=0;col<D;++col)
        if(ring[(pos%W)*D+col]!=word(raw[pos*D+col])) std::exit(6);
      ++steps;
    }
    // Upstream cache failure exposes zero output and cannot mutate the ring.
    std::vector<unsigned short> before(ring,ring+W*D);
    counts[3]=0; reset[0]=0; cache_append[2]=1; execute();
    if(!append[2] || !history[1] || std::memcmp(before.data(),ring,W*D*2)) std::exit(7);
    for(int i=0;i<R*H*D;++i) if(word(__bfloat162float(output[i]))!=0) std::exit(8);
    ++steps;
  }
  check(cudaFree(sink)); check(cudaFree(ring)); check(cudaFree(output)); check(cudaFree(compressed));
  check(cudaFree(current)); check(cudaFree(queries)); check(cudaFree(append)); check(cudaFree(history));
  check(cudaFree(offsets)); check(cudaFree(ids)); check(cudaFree(selected_count));
  check(cudaFree(cache_append)); check(cudaFree(cache_count)); check(cudaFree(positions)); check(cudaFree(reset)); check(cudaFree(counts));
  std::printf("joint mode=%d steps=%d causal-union=passed ring=exact inactive=zero release=balanced\n",Mode,steps);
}
int main() {
  run<4,8,4,32,4,3>(); run<128,8,4,256,128,0>(); run<0,3,3,3,4,0>();
  std::puts("joint attention component correctness passed"); return 0;
}
