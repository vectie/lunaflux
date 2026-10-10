// Independent CPU retained-ring/noncausal-draft oracle. No whole-model claim.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "main.cuh"

static void check(cudaError_t r) {
  if (r != cudaSuccess) { std::fprintf(stderr, "%s\n", cudaGetErrorString(r)); std::exit(2); }
}
template<class T> static T *allocate(size_t n) {
  T *p = nullptr; check(cudaMallocManaged(&p, n * sizeof(T))); return p;
}
static float b(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }
static float value(int token, int column) { return b(float((token * 7 + column * 3) % 29 - 14) * .125f); }
int main() {
  constexpr int width=8, heads=2, window=4, rows=3, main_rows=8;
  auto mc=allocate<int>(5), dc=allocate<int>(5), reset=allocate<int>(1), start=allocate<int>(1);
  auto history=allocate<int>(2), descriptor=allocate<int>(4);
  auto main_kv=allocate<__nv_bfloat16>(main_rows*width), draft=allocate<__nv_bfloat16>(rows*width);
  auto query=allocate<__nv_bfloat16>(rows*heads*width), ring=allocate<__nv_bfloat16>(window*width);
  auto output=allocate<__nv_bfloat16>(rows*heads*width);
  auto sink=allocate<float>(heads);
  std::vector<float> expected_ring(window*width), expected(rows*heads*width);
  std::vector<__nv_bfloat16> first(rows*heads*width), saved_ring(window*width);
  int cases=0; float worst=0;
  for(int h=0;h<heads;++h) sink[h]=h ? .25f : -.5f;
  auto submit=[&](bool prime) {
    if(prime) {
      prime_reserve<<<1,1>>>(mc,dc,reset,start,history,descriptor);
      prime_copy<<<window,128>>>(descriptor,(uint16_t*)main_kv,(uint16_t*)ring);
      prime_publish<<<1,1>>>(descriptor,history);
    } else {
      predict_reserve<<<1,1>>>(mc,dc,reset,start,history,descriptor);
      predict_copy<<<window,128>>>(descriptor,(uint16_t*)main_kv,(uint16_t*)ring);
      predict_publish<<<1,1>>>(descriptor,history);
      predict_attention<<<dim3(rows,heads),128>>>(descriptor,query,draft,ring,sink,output);
    }
    check(cudaGetLastError()); check(cudaDeviceSynchronize());
  };
  auto sentinel=[&] { for(int i=0;i<rows*heads*width;++i) output[i]=__float2bfloat16_rn(19.f); };
  for(int primed : {1,3,4,5,8}) {
    std::memset(history,0,2*sizeof(int)); std::memset(mc,0,5*sizeof(int)); std::memset(dc,0,5*sizeof(int));
    mc[2]=1; mc[3]=primed; dc[2]=1; dc[3]=rows; reset[0]=1; start[0]=0;
    std::fill(expected_ring.begin(),expected_ring.end(),23.f);
    for(int i=0;i<window*width;++i) ring[i]=__float2bfloat16_rn(23.f);
    for(int row=0;row<main_rows;++row) for(int c=0;c<width;++c) main_kv[row*width+c]=__float2bfloat16_rn(value(row,c));
    for(int absolute=0;absolute<primed;++absolute) for(int c=0;c<width;++c) expected_ring[(absolute%window)*width+c]=value(absolute,c);
    sentinel(); submit(true);
    if(history[0]!=primed || history[1] || descriptor[3]) return 3;
    for(int i=0;i<window*width;++i) if(__bfloat162float(ring[i])!=expected_ring[i]) return 4;
    for(int i=0;i<rows*heads*width;++i) if(__bfloat162float(output[i])!=19.f) return 5;
    ++cases;
    for(int position=primed;position<25;++position) {
      reset[0]=0; start[0]=position; mc[3]=1; dc[3]=1+(position%rows);
      for(int c=0;c<width;++c) { main_kv[c]=__float2bfloat16_rn(value(position,c)); expected_ring[(position%window)*width+c]=value(position,c); }
      for(int row=0;row<rows;++row) for(int c=0;c<width;++c) draft[row*width+c]=__float2bfloat16_rn(value(100+position+row,c));
      for(int i=0;i<rows*heads*width;++i) query[i]=__float2bfloat16_rn(float((i*5+position)%11-5)*.0625f);
      std::fill(expected.begin(),expected.end(),19.f);
      const int length=std::min(window,position+1)+dc[3], main=std::min(window,position+1);
      for(int row=0;row<dc[3];++row) for(int h=0;h<heads;++h) {
        std::vector<float> scores(length); float maximum=sink[h];
        for(int slot=0;slot<length;++slot) {
          float dot=0;
          for(int c=0;c<width;++c) {
            float kv=slot<main ? expected_ring[slot*width+c] : __bfloat162float(draft[(slot-main)*width+c]);
            dot=dot+__bfloat162float(query[(row*heads+h)*width+c])*kv;
          }
          scores[slot]=dot*(1.f/std::sqrt(float(width))); maximum=std::max(maximum,scores[slot]);
        }
        float denominator=std::exp(sink[h]-maximum);
        for(float &score:scores) { score=std::exp(score-maximum); denominator=denominator+score; }
        for(int c=0;c<width;++c) {
          float sum=0;
          for(int slot=0;slot<length;++slot) {
            float kv=slot<main ? expected_ring[slot*width+c] : __bfloat162float(draft[(slot-main)*width+c]);
            sum=sum+scores[slot]*kv;
          }
          expected[(row*heads+h)*width+c]=b(sum/denominator);
        }
      }
      // Repeat identical request state, not an illegal second committed append.
      for(int repeat=0;repeat<2;++repeat) {
        history[0]=position; history[1]=0; sentinel(); submit(false);
        if(history[0]!=position+1 || history[1] || descriptor[3]) return 6;
        for(int i=0;i<window*width;++i) if(__bfloat162float(ring[i])!=expected_ring[i]) return 7;
        for(int i=0;i<rows*heads*width;++i) {
          float error=std::fabs(__bfloat162float(output[i])-expected[i]); worst=std::max(worst,error);
          if(!std::isfinite(__bfloat162float(output[i])) || error>.00390625f+.005f*std::fabs(expected[i])) {
            std::fprintf(stderr,"position=%d index=%d got=%g expected=%g\n",position,i,__bfloat162float(output[i]),expected[i]); return 8;
          }
        }
        if(!repeat) std::memcpy(first.data(),output,first.size()*sizeof(__nv_bfloat16));
        else if(std::memcmp(first.data(),output,first.size()*sizeof(__nv_bfloat16))) return 9;
        ++cases;
      }
      // Prediction from a published prefix must match the append-and-predict
      // result without republishing KV, resetting or advancing its frontier.
      std::memcpy(saved_ring.data(),ring,saved_ring.size()*sizeof(__nv_bfloat16));
      for(int retained_reset : {0,1}) {
        reset[0]=retained_reset; sentinel();
        committed_reserve<<<1,1>>>(mc,dc,reset,start,history,descriptor);
        committed_attention<<<dim3(rows,heads),128>>>(descriptor,query,draft,ring,sink,output);
        check(cudaGetLastError()); check(cudaDeviceSynchronize());
        if(history[0]!=position+1 || history[1] || descriptor[3] ||
           std::memcmp(saved_ring.data(),ring,saved_ring.size()*sizeof(__nv_bfloat16)) ||
           std::memcmp(first.data(),output,first.size()*sizeof(__nv_bfloat16))) return 30;
        ++cases;
      }
    }
    // Malformed metadata cannot mutate the ring or output; error is sticky.
    for(int invalid=0;invalid<8;++invalid) {
      history[0]=25; history[1]=0; start[0]=25; reset[0]=0; mc[2]=1; mc[3]=1; dc[2]=1; dc[3]=3;
      if(invalid==0) mc[3]=2;
      if(invalid==1) dc[3]=0;
      if(invalid==2) dc[3]=4;
      if(invalid==3) start[0]=26;
      if(invalid==4) reset[0]=2;
      if(invalid==5) mc[2]=2;
      if(invalid==6) dc[2]=2;
      if(invalid==7) start[0]=31;
      std::memcpy(saved_ring.data(),ring,saved_ring.size()*sizeof(__nv_bfloat16));
      sentinel(); submit(false);
      if(!history[1] || !descriptor[3] || std::memcmp(saved_ring.data(),ring,saved_ring.size()*sizeof(__nv_bfloat16))) return 10;
      for(int i=0;i<rows*heads*width;++i) if(__bfloat162float(output[i])!=19.f) return 11;
      start[0]=25; reset[0]=0; mc[2]=1; mc[3]=1; dc[2]=1; dc[3]=3;
      submit(false); if(!history[1] || !descriptor[3]) return 12;
      ++cases;
    }
  }
  // Invalid committed views neither poison the retained prefix nor emit output.
  history[0]=12; history[1]=0; start[0]=9; mc[2]=1; mc[3]=3; dc[2]=1; dc[3]=3; reset[0]=0;
  for(int invalid=0;invalid<5;++invalid) {
    start[0]=9; mc[3]=3; dc[3]=3; reset[0]=0;
    if(invalid==0) start[0]=8;
    if(invalid==1) mc[3]=0;
    if(invalid==2) dc[3]=0;
    if(invalid==3) reset[0]=2;
    if(invalid==4) mc[3]=9;
    sentinel();
    committed_reserve<<<1,1>>>(mc,dc,reset,start,history,descriptor);
    committed_attention<<<dim3(rows,heads),128>>>(descriptor,query,draft,ring,sink,output);
    check(cudaGetLastError()); check(cudaDeviceSynchronize());
    if(history[0]!=12 || history[1] || !descriptor[3]) return 31;
    for(int i=0;i<rows*heads*width;++i) if(__bfloat162float(output[i])!=19.f) return 32;
    ++cases;
  }
  // Later prompt chunks and verified-prefix replay append at the frontier.
  history[0]=0; history[1]=0; mc[2]=1; dc[2]=1;
  std::fill(expected_ring.begin(),expected_ring.end(),23.f);
  for(int i=0;i<window*width;++i) ring[i]=__float2bfloat16_rn(23.f);
  int frontier=0;
  for(int count : {3,2,8,7,5}) {
    reset[0]=frontier==0; start[0]=frontier; mc[3]=count;
    for(int row=0;row<count;++row) for(int c=0;c<width;++c) {
      main_kv[row*width+c]=__float2bfloat16_rn(value(frontier+row,c));
      expected_ring[((frontier+row)%window)*width+c]=value(frontier+row,c);
    }
    sentinel(); submit(true); frontier+=count;
    if(history[0]!=frontier || history[1] || descriptor[3]) return 14;
    for(int i=0;i<window*width;++i) if(__bfloat162float(ring[i])!=expected_ring[i]) return 15;
    for(int i=0;i<rows*heads*width;++i) if(__bfloat162float(output[i])!=19.f) return 16;
    ++cases;
    // Multi-row prompt/accepted-prefix commits use the last committed position
    // without replaying any main projection or rewriting retained ring slots.
    dc[3]=rows;
    for(int row=0;row<rows;++row) for(int c=0;c<width;++c)
      draft[row*width+c]=__float2bfloat16_rn(value(100+frontier+row,c));
    for(int i=0;i<rows*heads*width;++i)
      query[i]=__float2bfloat16_rn(float((i*5+frontier)%11-5)*.0625f);
    const int retained=std::min(window,frontier), length=retained+rows;
    for(int row=0;row<rows;++row) for(int h=0;h<heads;++h) {
      std::vector<float> scores(length); float maximum=sink[h];
      for(int slot=0;slot<length;++slot) {
        float dot=0;
        for(int c=0;c<width;++c) {
          const float kv=slot<retained ? expected_ring[slot*width+c] : __bfloat162float(draft[(slot-retained)*width+c]);
          dot=dot+__bfloat162float(query[(row*heads+h)*width+c])*kv;
        }
        scores[slot]=dot*(1.f/std::sqrt(float(width))); maximum=std::max(maximum,scores[slot]);
      }
      float denominator=std::exp(sink[h]-maximum);
      for(float &score:scores) { score=std::exp(score-maximum); denominator=denominator+score; }
      for(int c=0;c<width;++c) {
        float sum=0;
        for(int slot=0;slot<length;++slot) {
          const float kv=slot<retained ? expected_ring[slot*width+c] : __bfloat162float(draft[(slot-retained)*width+c]);
          sum=sum+scores[slot]*kv;
        }
        expected[(row*heads+h)*width+c]=b(sum/denominator);
      }
    }
    std::memcpy(saved_ring.data(),ring,saved_ring.size()*sizeof(__nv_bfloat16));
    sentinel();
    committed_reserve<<<1,1>>>(mc,dc,reset,start,history,descriptor);
    committed_attention<<<dim3(rows,heads),128>>>(descriptor,query,draft,ring,sink,output);
    check(cudaGetLastError()); check(cudaDeviceSynchronize());
    if(history[0]!=frontier || history[1] || descriptor[3] || descriptor[0]!=frontier-1 ||
       std::memcmp(saved_ring.data(),ring,saved_ring.size()*sizeof(__nv_bfloat16))) return 33;
    for(int i=0;i<rows*heads*width;++i) {
      const float error=std::fabs(__bfloat162float(output[i])-expected[i]); worst=std::max(worst,error);
      if(!std::isfinite(__bfloat162float(output[i])) || error>.00390625f+.005f*std::fabs(expected[i])) return 34;
    }
    ++cases;
  }
  // Undo restores overwritten physical slots, not just the logical length.
  auto backup_ring=allocate<__nv_bfloat16>(window*width);
  auto backup_history=allocate<int>(2);
  check(cudaMemcpy(backup_ring,ring,window*width*sizeof(__nv_bfloat16),cudaMemcpyDeviceToDevice));
  check(cudaMemcpy(backup_history,history,2*sizeof(int),cudaMemcpyDeviceToDevice));
  reset[0]=0; start[0]=frontier; mc[3]=4;
  for(int row=0;row<4;++row) for(int c=0;c<width;++c) main_kv[row*width+c]=__float2bfloat16_rn(value(frontier+row,c));
  submit(true); if(history[0]!=frontier+4 || history[1]) return 17;
  bool changed=false;
  for(int i=0;i<window*width;++i) changed|=__bfloat162float(ring[i])!=expected_ring[i];
  if(!changed) return 18;
  check(cudaMemcpy(ring,backup_ring,window*width*sizeof(__nv_bfloat16),cudaMemcpyDeviceToDevice));
  check(cudaMemcpy(history,backup_history,2*sizeof(int),cudaMemcpyDeviceToDevice));
  if(history[0]!=frontier || history[1]) return 19;
  for(int i=0;i<window*width;++i) if(__bfloat162float(ring[i])!=expected_ring[i]) return 20;
  ++cases;
  // Replay only the accepted two rows after restoring the tentative four.
  start[0]=frontier; mc[3]=2; sentinel(); submit(true);
  for(int row=0;row<2;++row) for(int c=0;c<width;++c) expected_ring[((frontier+row)%window)*width+c]=value(frontier+row,c);
  frontier+=2;
  if(history[0]!=frontier || history[1]) return 21;
  for(int i=0;i<window*width;++i) if(__bfloat162float(ring[i])!=expected_ring[i]) return 22;
  for(int i=0;i<rows*heads*width;++i) if(__bfloat162float(output[i])!=19.f) return 23;
  ++cases;
  // A gap remains illegal; general append retains frontier ownership.
  start[0]=frontier+1; mc[3]=1; submit(true);
  if(history[0]!=frontier || !history[1] || !descriptor[3]) return 24;
  for(int i=0;i<window*width;++i) if(__bfloat162float(ring[i])!=expected_ring[i]) return 25;
  ++cases;
  check(cudaFree(backup_history)); check(cudaFree(backup_ring));
  // Future draft visibility: row zero with zero queries sees the last KV row.
  history[0]=1; history[1]=0; mc[3]=1; dc[3]=3; reset[0]=0; start[0]=1;
  for(int i=0;i<rows*heads*width;++i) query[i]=__float2bfloat16_rn(0.f);
  for(int i=0;i<rows*width;++i) draft[i]=__float2bfloat16_rn(0.f);
  submit(false); float before=__bfloat162float(output[0]);
  history[0]=1; draft[2*width]=__float2bfloat16_rn(16.f); submit(false);
  if(__bfloat162float(output[0])-before<1.f) return 13;
  ++cases;
  check(cudaFree(sink)); check(cudaFree(output)); check(cudaFree(ring)); check(cudaFree(query)); check(cudaFree(draft)); check(cudaFree(main_kv));
  check(cudaFree(descriptor)); check(cudaFree(history)); check(cudaFree(start)); check(cudaFree(reset)); check(cudaFree(dc)); check(cudaFree(mc));
  check(cudaDeviceReset());
  std::printf("committed draft: %d cases passed; maxabs=%g; chunk-prime/wrap/device-undo/accepted-prefix-replay/all-draft visibility/sticky failure; allocations released\n",cases,worst);
}
