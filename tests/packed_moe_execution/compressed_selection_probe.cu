#include "compressed_selection_kernels.cuh"
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <vector>

#define CK(expr) do { const cudaError_t e = (expr); if (e != cudaSuccess) { \
  std::fprintf(stderr, "%s: %s\n", #expr, cudaGetErrorString(e)); std::exit(2); \
} } while (0)

uint16_t bf16(float value) {
  uint32_t bits; std::memcpy(&bits, &value, 4);
  if ((bits & 0x7fffffffU) > 0x7f800000U) return uint16_t((bits >> 16) | 0x40U);
  return uint16_t((bits + 0x7fffU + ((bits >> 16) & 1U)) >> 16);
}
float fp32(uint16_t value) {
  const uint32_t bits = uint32_t(value) << 16; float result;
  std::memcpy(&result, &bits, 4); return result;
}
float add(float a, float b) { volatile float result = a + b; return result; }
float mul(float a, float b) { volatile float result = a * b; return result; }
template<class T> T *upload(const std::vector<T>& data) {
  T *p; CK(cudaMalloc(&p, data.size() * sizeof(T)));
  CK(cudaMemcpy(p, data.data(), data.size() * sizeof(T), cudaMemcpyHostToDevice)); return p;
}
template<class T> void write(T *p, const std::vector<T>& data) {
  CK(cudaMemcpy(p, data.data(), data.size() * sizeof(T), cudaMemcpyHostToDevice));
}
template<class T> void read(std::vector<T>& data, T *p) {
  CK(cudaMemcpy(data.data(), p, data.size() * sizeof(T), cudaMemcpyDeviceToHost));
}

void test(int capacity) {
  constexpr int rows=3, heads=2, width=128, ratio=4;
  const int physical=std::max(1, capacity), slots=std::max(1, std::min(8, capacity));
  const int maximum_positions=capacity ? 1040 : 3;
  std::vector<uint16_t> queries(rows*heads*width), weights(rows*heads), keys(physical*width), scores(rows*physical);
  for (size_t i=0; i<queries.size(); ++i) queries[i]=bf16(float(int(i%23)-11)/32);
  for (size_t i=0; i<keys.size(); ++i) keys[i]=bf16(float(int(i%31)-15)/16);
  for (size_t i=0; i<weights.size(); ++i) weights[i]=bf16(i%2 ? -0.1875f : 0.3125f);
  std::vector<int32_t> counts(5), positions(rows), cache_counts(5), append(3), offsets(rows), selected_counts(rows), selected(rows*slots);
  auto q=upload(queries), w=upload(weights), k=upload(keys), s=upload(scores);
  auto c=upload(counts), p=upload(positions), cc=upload(cache_counts), a=upload(append), o=upload(offsets), sc=upload(selected_counts), si=upload(selected);
  int cases=0;
  auto run = [&](int live, int retained, int owner, int error, int offset, bool tie, int bad_position=-2) {
    counts[2]=1; counts[3]=live; cache_counts[2]=owner; cache_counts[3]=retained; append[2]=error;
    positions = capacity ? std::vector<int32_t>{3, 7, 1039} : std::vector<int32_t>{0,1,2};
    if (bad_position!=-2) positions[2]=bad_position;
    for (int row=0; row<rows; ++row) offsets[row]=offset;
    if (tie) std::fill(queries.begin(), queries.end(), bf16(0));
    write(q,queries); write(c,counts); write(p,positions); write(cc,cache_counts); write(a,append); write(o,offsets);
    if (capacity) selection260_scores<<<dim3(rows,2),256>>>(c,p,cc,a,(__nv_bfloat16*)q,(__nv_bfloat16*)w,(__nv_bfloat16*)k,(__nv_bfloat16*)s);
    else selection0_scores<<<dim3(rows,1),256>>>(c,p,cc,a,(__nv_bfloat16*)q,(__nv_bfloat16*)w,(__nv_bfloat16*)k,(__nv_bfloat16*)s);
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize()); read(scores,s);
    std::vector<uint16_t> expected_scores(rows*physical,bf16(-INFINITY));
    const bool valid=live>0 && live<=rows && !error && retained>=0 && retained<=capacity && (!retained || owner==1);
    for (int row=0; valid && row<live; ++row) {
      if (positions[row]<0 || positions[row]>=maximum_positions) continue;
      for (int candidate=0; candidate<std::min(retained,(positions[row]+1)/ratio); ++candidate) {
        float total=0;
        for (int head=0; head<heads; ++head) {
          float dot=0;
          for (int col=0; col<width; ++col) dot=add(dot,mul(fp32(queries[(row*heads+head)*width+col]),fp32(keys[candidate*width+col])));
          const float projected=fp32(bf16(dot));
          const float weighted=fp32(bf16(mul(std::max(projected,0.0f),fp32(weights[row*heads+head]))));
          total=add(total,weighted);
        }
        expected_scores[row*physical+candidate]=bf16(total);
      }
    }
    if (scores!=expected_scores) { std::fprintf(stderr,"score publication mismatch capacity=%d case=%d\n",capacity,cases); std::exit(3); }
    if (capacity) selection260_select<<<rows,256>>>(c,p,cc,a,o,(__nv_bfloat16*)s,sc,si);
    else selection0_select<<<rows,256>>>(c,p,cc,a,o,(__nv_bfloat16*)s,sc,si);
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize()); read(selected_counts,sc); read(selected,si);
    std::vector<int32_t> expected_counts(rows), expected(rows*slots,-1);
    const bool valid_offset=offset>=0 && (!retained || int64_t(offset)+retained-1<=INT_MAX);
    for (int row=0; valid && valid_offset && row<live; ++row) {
      if (positions[row]<0 || positions[row]>=maximum_positions) continue;
      const int visible=std::min(retained,(positions[row]+1)/ratio);
      std::vector<int> candidates(visible); std::iota(candidates.begin(),candidates.end(),0);
      std::stable_sort(candidates.begin(),candidates.end(),[&](int x,int y) { return fp32(expected_scores[row*physical+x])>fp32(expected_scores[row*physical+y]); });
      expected_counts[row]=std::min(visible, capacity ? slots : 0);
      for (int rank=0; rank<expected_counts[row]; ++rank) expected[row*slots+rank]=offset+candidates[rank];
    }
    if (selected!=expected || selected_counts!=expected_counts) { std::fprintf(stderr,"causal topk mismatch capacity=%d case=%d\n",capacity,cases); std::exit(4); }
    ++cases;
  };
  run(3,capacity,capacity?1:0,0,17,false);
  run(1,capacity,capacity?1:0,0,0,false);
  run(0,capacity,capacity?1:0,0,0,false);
  run(3,0,0,0,0,false);
  run(3,capacity,capacity?1:0,1,0,false);
  run(3,capacity+1,1,0,0,false);
  run(3,-1,1,0,0,false);
  run(4,capacity,1,0,0,false);
  run(3,capacity,0,0,0,false);
  run(3,capacity,capacity?1:0,0,-1,false);
  run(3,capacity,capacity?1:0,0,INT_MAX,false);
  run(3,capacity,capacity?1:0,0,3,true);
  run(3,capacity,capacity?1:0,0,3,true);
  run(3,capacity,capacity?1:0,0,3,true,-1);
  run(3,capacity,capacity?1:0,0,3,true,maximum_positions);
  for (void *ptr : {(void*)q,(void*)w,(void*)k,(void*)s,(void*)c,(void*)p,(void*)cc,(void*)a,(void*)o,(void*)sc,(void*)si}) CK(cudaFree(ptr));
  std::printf("selection capacity=%d cases=%d bf16_publications=passed causal_topk=passed replay=passed\n",capacity,cases);
}
int main() { test(260); test(0); return 0; }
