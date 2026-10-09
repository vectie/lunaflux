// Tiny CPU-oracle fixture for the generated precision-IR text ingress/egress.
// This is not a real checkpoint, decoder, model-quality or throughput test.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include "text_io_kernels.cuh"

static void check(cudaError_t error) {
  if (error != cudaSuccess) {
    std::fprintf(stderr, "CUDA: %s\n", cudaGetErrorString(error));
    std::exit(2);
  }
}

template<class T> static T* allocate(size_t n) {
  T* ptr = nullptr;
  check(cudaMallocManaged(&ptr, n*sizeof(T)));
  std::memset(ptr, 0, n*sizeof(T));
  return ptr;
}

static float round_bf(float x) {
  return __bfloat162float(__float2bfloat16_rn(x));
}

static bool same(__nv_bfloat16 a, __nv_bfloat16 b) {
  return std::memcmp(&a, &b, sizeof(a)) == 0;
}

int main() {
  constexpr int H=8, S=4, V=17, R=5, O=2;
  int32_t *counts=allocate<int32_t>(5), *ids=allocate<int32_t>(R);
  int32_t *output_count=allocate<int32_t>(1), *selected=allocate<int32_t>(O);
  int32_t *tokens=allocate<int32_t>(O);
  auto embedding=allocate<__nv_bfloat16>(V*H);
  auto residual=allocate<__nv_bfloat16>(R*S*H);
  auto norm_weight=allocate<__nv_bfloat16>(H);
  auto head=allocate<__nv_bfloat16>(V*H);
  auto hidden=allocate<__nv_bfloat16>(O*H);
  auto embedded=allocate<__nv_bfloat16>(R*S*H);
  auto logits=allocate<float>(O*V);
  counts[0]=1; counts[2]=1; counts[3]=R; output_count[0]=O;
  selected[0]=4; selected[1]=1;
  for(int r=0;r<R;++r) ids[r]=(r*3+2)%V;
  for(int i=0;i<V*H;++i) {
    embedding[i]=__float2bfloat16_rn(float((i*3)%19-9)*0.125f);
    head[i]=__float2bfloat16_rn(float((i*7)%23-11)*0.0625f);
  }
  for(int i=0;i<R*S*H;++i)
    residual[i]=__float2bfloat16_rn(float((i*5)%29-14)*0.125f);
  for(int c=0;c<H;++c) norm_weight[c]=__float2bfloat16_rn(0.5f+c*0.0625f);
  __nv_bfloat16 expected_hidden[O*H];
  float expected_logits[O*V];
  int expected_tokens[O];
  for(int row=0;row<O;++row) {
    float means[H], squares=0.0f;
    for(int c=0;c<H;++c) {
      float sum=0.0f;
      for(int s=0;s<S;++s)
        sum=sum+__bfloat162float(residual[(selected[row]*S+s)*H+c]);
      means[c]=round_bf(sum/float(S));
      squares=squares+means[c]*means[c];
    }
    const float inverse=1.0f/std::sqrt(squares/float(H)+1.0e-5f);
    for(int c=0;c<H;++c)
      expected_hidden[row*H+c]=__float2bfloat16_rn(
        round_bf(means[c]*inverse)*__bfloat162float(norm_weight[c]));
    float best=-INFINITY;
    expected_tokens[row]=-1;
    for(int w=0;w<V;++w) {
      float sum=0.0f;
      for(int c=0;c<H;++c)
        sum=sum+__bfloat162float(expected_hidden[row*H+c])*
                __bfloat162float(head[w*H+c]);
      expected_logits[row*V+w]=round_bf(sum);
      if(expected_tokens[row]<0 || expected_logits[row*V+w]>best) {
        best=expected_logits[row*V+w]; expected_tokens[row]=w;
      }
    }
  }
  __nv_bfloat16 first_hidden[O*H];
  float first_logits[O*V];
  for(int repeat=0;repeat<2;++repeat) {
    text_fixture_embed<<<1,256>>>(counts, ids, embedding, embedded);
    text_fixture_norm<<<O,256>>>(counts, output_count, selected, residual, norm_weight, hidden);
    text_fixture_head<<<1,256>>>(output_count, hidden, head, logits);
    text_fixture_greedy<<<O,1>>>(counts, output_count, selected, logits, tokens);
    check(cudaGetLastError()); check(cudaDeviceSynchronize());
    for(int i=0;i<R*S*H;++i)
      if(!same(embedded[i], embedding[ids[i/(S*H)]*H+i%H])) return 3;
    float norm_error=0.0f, head_error=0.0f;
    for(int i=0;i<O*H;++i)
      norm_error=std::max(norm_error, std::fabs(__bfloat162float(hidden[i])-
                                             __bfloat162float(expected_hidden[i])));
    for(int i=0;i<O*V;++i)
      head_error=std::max(head_error,std::fabs(logits[i]-expected_logits[i]));
    if(norm_error>0.015625f || head_error>0.03125f) return 4;
    for(int i=0;i<O;++i) if(tokens[i]!=expected_tokens[i]) return 5;
    if(repeat==0) {
      std::memcpy(first_hidden, hidden, sizeof(first_hidden));
      std::memcpy(first_logits, logits, sizeof(first_logits));
    } else if(std::memcmp(first_hidden, hidden, sizeof(first_hidden)) ||
              std::memcmp(first_logits, logits, sizeof(first_logits))) return 6;
    std::printf("repeat=%d embed=bitwise norm_maxabs=%.9g head_maxabs=%.9g greedy=exact\n",
                repeat, norm_error, head_error);
  }
  // Ties select the first vocabulary entry; non-finite logits fail explicitly.
  for(int i=0;i<O*V;++i) logits[i]=1.0f;
  text_fixture_greedy<<<O,1>>>(counts,output_count,selected,logits,tokens);
  check(cudaDeviceSynchronize());
  if(tokens[0]!=0 || tokens[1]!=0) return 7;
  logits[3]=NAN;
  selected[1]=R;
  text_fixture_greedy<<<O,1>>>(counts,output_count,selected,logits,tokens);
  check(cudaDeviceSynchronize());
  if(tokens[0]!=-1 || tokens[1]!=-1) return 8;
  // A chunk producing no token must not touch output ports.
  output_count[0]=0; tokens[0]=123; tokens[1]=124;
  text_fixture_greedy<<<O,1>>>(counts,output_count,selected,logits,tokens);
  check(cudaDeviceSynchronize());
  if(tokens[0]!=123 || tokens[1]!=124) return 9;
  check(cudaFree(logits)); check(cudaFree(embedded)); check(cudaFree(hidden));
  check(cudaFree(head)); check(cudaFree(norm_weight)); check(cudaFree(residual));
  check(cudaFree(embedding)); check(cudaFree(tokens)); check(cudaFree(selected));
  check(cudaFree(output_count)); check(cudaFree(ids)); check(cudaFree(counts));
  std::puts("text_io_correctness=passed geometry=R5,H8,S4,V17,O2 deterministic=true");
  return 0;
}
