// Independent numerical probe for complete-head producer packing. Include an
// exported production source with -DKERNEL_SOURCE='"/absolute/source.cu"'.
// Distinct segment/head/component weights catch ownership and tail mistakes;
// exact dyadic projection inputs avoid obscuring them with GEMM rounding noise.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include KERNEL_SOURCE
#ifndef PROBE_MAX_TOKENS
#define PROBE_MAX_TOKENS 129
#endif
#define CK(call) do { cudaError_t e = (call); if (e != cudaSuccess) { std::fprintf(stderr, "%s\n", cudaGetErrorString(e)); std::exit(2); } } while(0)
struct Buffer {
  void *p;
  explicit Buffer(size_t bytes) { CK(cudaMalloc(&p, bytes)); CK(cudaMemset(p, 0, bytes)); }
  ~Buffer() { CK(cudaFree(p)); }
  template<class T> T* as() { return static_cast<T*>(p); }
  void put(const void *h, size_t n) { CK(cudaMemcpy(p, h, n, cudaMemcpyHostToDevice)); }
};
static float round_bf16(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }
static uint64_t fold_bytes(uint64_t hash, const std::vector<__nv_bfloat16>& values) {
  const auto* bytes = reinterpret_cast<const unsigned char*>(values.data());
  for (size_t i=0; i<values.size()*sizeof(__nv_bfloat16); ++i) {
    hash ^= bytes[i]; hash *= 1099511628211ULL;
  }
  return hash;
}
static std::vector<__nv_bfloat16> weights(int segment, int heads) {
  std::vector<__nv_bfloat16> out(heads * LF_HEAD_DIM * LF_INPUT_WIDTH);
  for (int h = 0; h < heads; ++h) for (int c = 0; c < LF_HEAD_DIM; ++c)
    for (int k = 0; k < LF_INPUT_WIDTH; ++k) {
      const float numerator = float(1 + segment * 3 + h % 3 + c % 5);
      const float sign = ((c + k) % 7 == 0) ? -1.0f : 1.0f;
      out[(h * LF_HEAD_DIM + c) * LF_INPUT_WIDTH + k] =
        __float2bfloat16_rn(sign * numerator / LF_INPUT_WIDTH);
    }
  return out;
}
int main() {
  CK(cudaSetDevice(0));
  for (int tokens : {1,2,7,8,15,16,17,31,32,33,63,64,65,127,128,129}) {
    if (tokens > PROBE_MAX_TOKENS) continue;
    const int page_count = (tokens + LF_TOKENS_PER_PAGE - 1) / LF_TOKENS_PER_PAGE;
    int counts[5] = {1,0,1,tokens,page_count}, offsets[2] = {0,tokens}, tables[2] = {0,page_count};
    std::vector<int> pages(page_count), positions(tokens);
    for (int p = 0; p < page_count; ++p) pages[p] = page_count - p - 1;
    for (int t = 0; t < tokens; ++t) positions[t] = t;
    Buffer dc(sizeof counts), dpos(tokens*4), doff(8), dseq(4), dtoff(8), dpage(page_count*4);
    dc.put(counts,sizeof counts); dpos.put(positions.data(),tokens*4); doff.put(offsets,8);
    dseq.put(&tokens,4); dtoff.put(tables,8); dpage.put(pages.data(),page_count*4);
    auto q = weights(0, LF_QUERY_HEADS), k = weights(1, LF_KV_HEADS), v = weights(2, LF_KV_HEADS);
    std::vector<__nv_bfloat16> x(tokens*LF_INPUT_WIDTH), qnorm(LF_HEAD_DIM), knorm(LF_HEAD_DIM);
    for (int t = 0; t < tokens; ++t) for (int i = 0; i < LF_INPUT_WIDTH; ++i)
      x[t*LF_INPUT_WIDTH+i] = __float2bfloat16_rn(float((t%7+1)*(i%4+1))/32.0f);
    for (int c = 0; c < LF_HEAD_DIM; ++c) {
      qnorm[c] = __float2bfloat16_rn(1.0f + float(c%3)/8.0f);
      knorm[c] = __float2bfloat16_rn(0.5f + float(c%5)/8.0f);
    }
    Buffer dx(x.size()*2), dq(q.size()*2), dwk(k.size()*2), dwv(v.size()*2);
    Buffer dqn(qnorm.size()*2), dkn(knorm.size()*2), dout(tokens*LF_OUTPUT_WIDTH*2);
    const size_t arena_elements = size_t(LF_TOTAL_PAGE_COUNT)*LF_PAGE_STRIDE_BF16;
    Buffer dk(arena_elements*2), dv(arena_elements*2);
    dx.put(x.data(),x.size()*2); dq.put(q.data(),q.size()*2); dwk.put(k.data(),k.size()*2);
    dwv.put(v.data(),v.size()*2); dqn.put(qnorm.data(),qnorm.size()*2); dkn.put(knorm.data(),knorm.size()*2);
    lunaflux_fused_qwen_qkv_qknorm_rope_kvwrite_bf16_head128_production_v2<<<dim3((tokens+LF_TILE_ROWS-1)/LF_TILE_ROWS,LF_HEAD_GROUPS),128>>>(
      dc.as<int>(),dpos.as<int>(),doff.as<int>(),dseq.as<int>(),dtoff.as<int>(),dpage.as<int>(),
      dx.as<__nv_bfloat16>(),dq.as<__nv_bfloat16>(),dwk.as<__nv_bfloat16>(),dwv.as<__nv_bfloat16>(),
      dqn.as<__nv_bfloat16>(),dkn.as<__nv_bfloat16>(),dout.as<__nv_bfloat16>(),dk.as<__nv_bfloat16>(),dv.as<__nv_bfloat16>());
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    std::vector<__nv_bfloat16> out(tokens*LF_OUTPUT_WIDTH), key(arena_elements), value(arena_elements);
    CK(cudaMemcpy(out.data(),dout.p,out.size()*2,cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(key.data(),dk.p,key.size()*2,cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(value.data(),dv.p,value.size()*2,cudaMemcpyDeviceToHost));
    std::vector<bool> touched(arena_elements, false);
    for (int t = 0; t < tokens; ++t) for (int h = 0; h < LF_PACKED_HEADS; ++h) {
      const auto& w = h < LF_QUERY_HEADS ? q : (h < LF_QK_HEADS ? k : v);
      const int local_head = h < LF_QUERY_HEADS ? h : (h < LF_QK_HEADS ? h-LF_QUERY_HEADS : h-LF_QK_HEADS);
      std::vector<float> ref(LF_HEAD_DIM);
      float squares = 0.0f;
      for (int c = 0; c < LF_HEAD_DIM; ++c) {
        float dot = 0.0f;
        for (int i = 0; i < LF_INPUT_WIDTH; ++i)
          dot += __bfloat162float(x[t*LF_INPUT_WIDTH+i]) * __bfloat162float(w[(local_head*LF_HEAD_DIM+c)*LF_INPUT_WIDTH+i]);
        ref[c] = round_bf16(dot); squares += ref[c] * ref[c];
      }
      if (h < LF_QK_HEADS) {
        const auto& norm = h < LF_QUERY_HEADS ? qnorm : knorm;
        const float inv = 1.0f/std::sqrt(squares/LF_HEAD_DIM + LF_NORM_EPSILON);
        for (int c = 0; c < LF_HEAD_DIM; ++c) ref[c] = round_bf16((ref[c]*inv)*__bfloat162float(norm[c]));
        for (int pair = 0; pair < LF_HEAD_DIM/2; ++pair) {
          const float angle = float(t)*float(std::pow(1000000.0,-2.0*pair/LF_HEAD_DIM));
          const float a=ref[pair], b=ref[pair+LF_HEAD_DIM/2];
          ref[pair] = round_bf16(a*std::cos(angle)-b*std::sin(angle));
          ref[pair+LF_HEAD_DIM/2] = round_bf16(b*std::cos(angle)+a*std::sin(angle));
        }
      }
      for (int c = 0; c < LF_HEAD_DIM; ++c) {
        const float got=__bfloat162float(out[t*LF_OUTPUT_WIDTH+h*LF_HEAD_DIM+c]);
        if (!std::isfinite(got) || std::fabs(got-ref[c]) > 0.02f) {
          std::fprintf(stderr,"mismatch d=%d heads=%d t=%d h=%d c=%d got=%g ref=%g\n",LF_HEAD_DIM,LF_HEADS_PER_CTA,t,h,c,got,ref[c]); return 3;
        }
        if (h >= LF_QUERY_HEADS) {
          const size_t index=size_t(pages[t/LF_TOKENS_PER_PAGE])*LF_PAGE_STRIDE_BF16+(t%LF_TOKENS_PER_PAGE)*LF_KV_WIDTH+local_head*LF_HEAD_DIM+c;
          touched[index]=true;
          if (__bfloat162float((h<LF_QK_HEADS?key:value)[index]) != got) return 4;
        }
      }
    }
    for (size_t i=0; i<arena_elements; ++i) if (!touched[i] && (__bfloat162float(key[i])!=0.0f || __bfloat162float(value[i])!=0.0f)) return 5;
    const uint64_t hash = fold_bytes(fold_bytes(fold_bytes(14695981039346656037ULL,out),key),value);
    std::printf("dimension=%d heads_per_cta=%d tokens=%d numeric=passed kv=passed tails=passed checksum=%016llx\n",LF_HEAD_DIM,LF_HEADS_PER_CTA,tokens,static_cast<unsigned long long>(hash));
  }
}
