#include "routing_kernels.cuh"
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

static void check(cudaError_t result) {
  if (result != cudaSuccess) {
    std::fprintf(stderr, "%s\n", cudaGetErrorString(result));
    std::exit(1);
  }
}
template <class T> static T *allocate(size_t elements) {
  T *result = nullptr;
  check(cudaMalloc(&result, elements * sizeof(T)));
  return result;
}
template <class T> static void upload(T *destination, const std::vector<T> &values) {
  check(cudaMemcpy(destination, values.data(), values.size()*sizeof(T), cudaMemcpyHostToDevice));
}
template <class T> static std::vector<T> download(T *source, size_t count) {
  std::vector<T> result(count);
  check(cudaMemcpy(result.data(), source, count*sizeof(T), cudaMemcpyDeviceToHost));
  return result;
}
static void equal(float actual, float expected, const char *stage) {
  if (!std::isfinite(actual) || std::fabs(actual-expected) > 2e-5f*(1+std::fabs(expected))) {
    std::fprintf(stderr, "%s mismatch %.9g %.9g\n", stage, actual, expected);
    std::exit(2);
  }
}

int main() {
  int *counts=allocate<int>(5), *indices=allocate<int>(4);
  __nv_bfloat16 *input=allocate<__nv_bfloat16>(64), *bf_weight=allocate<__nv_bfloat16>(128);
  float *weight=allocate<float>(128), *bias=allocate<float>(4), *logits=allocate<float>(8);
  float *scores=allocate<float>(8), *choice=allocate<float>(8), *weights=allocate<float>(4);
  int cases=0;
  for (bool bf16 : {true, false}) for (int pattern=0; pattern<3; ++pattern) for (int live=0; live<=2; ++live) {
    std::vector<__nv_bfloat16> x(64), wb(128);
    std::vector<float> w(128), b={0.0f,1.0f,0.0f,-1.0f};
    if (pattern==0) b.assign(4,0.0f);
    for (int i=0;i<64;++i) x[i]=__float2bfloat16(pattern==0 ? 0.0f : (i%7-3)*0.25f);
    for (int i=0;i<128;++i) {
      w[i]=pattern==2 ? (i/32%2 ? 40.0f : -40.0f) : (i/32-1)*0.125f;
      wb[i]=__float2bfloat16(w[i]);
    }
    upload(counts, std::vector<int>{0,0,1,live,0});
    upload(input,x); upload(weight,w); upload(bf_weight,wb); upload(bias,b);
    upload(logits,std::vector<float>(8,-17.0f));
    upload(scores,std::vector<float>(8,-17.0f));
    upload(choice,std::vector<float>(8,-17.0f));
    upload(indices,std::vector<int>(4,-7));
    upload(weights,std::vector<float>(4,-17.0f));
    if (bf16) {
      route_bf16_project<<<1,256>>>(counts,input,bf_weight,logits);
      route_bf16_score<<<1,256>>>(counts,logits,bias,scores,choice);
      route_bf16_choose<<<2,1>>>(counts,scores,choice,indices,weights);
    } else {
      route_f32_project<<<1,256>>>(counts,input,weight,logits);
      route_f32_score<<<1,256>>>(counts,logits,bias,scores,choice);
      route_f32_choose<<<2,1>>>(counts,scores,choice,indices,weights);
    }
    check(cudaGetLastError()); check(cudaDeviceSynchronize());
    auto l=download(logits,8), s=download(scores,8), c=download(choice,8), selected_weights=download(weights,4);
    auto selected_indices=download(indices,4);
    for (int row=0;row<2;++row) {
      float reference_scores[4], reference_choice[4];
      for (int e=0;e<4;++e) {
        float dot=0;
        for (int k=0;k<32;++k) {
          volatile float product=__bfloat162float(x[row*32+k])*(bf16 ? __bfloat162float(wb[e*32+k]) : w[e*32+k]);
          dot=dot+product;
        }
        float score=bf16 ? 1.0f/(1.0f+std::exp(-dot)) :
          std::sqrt(dot>0 ? dot+std::log1p(std::exp(-dot)) : std::log1p(std::exp(dot)));
        reference_scores[e]=score; reference_choice[e]=score+b[e];
        equal(l[row*4+e],row<live ? dot : -17.0f,"projection");
        equal(s[row*4+e],row<live ? score : -17.0f,"score");
        equal(c[row*4+e],row<live ? reference_choice[e] : -17.0f,"correction");
      }
      int group=std::max(reference_choice[0],reference_choice[1]) >= std::max(reference_choice[2],reference_choice[3]) ? 0 : 1;
      int a=group*2, z=a+1;
      if (reference_choice[z]>reference_choice[a]) std::swap(a,z);
      float denominator=reference_scores[a]+reference_scores[z]+1e-20f;
      for (int slot=0;slot<2;++slot) {
        int e=slot==0 ? a : z;
        if (selected_indices[row*2+slot] != (row<live ? e : -7)) {
          std::fprintf(stderr,"selected ID mismatch\n"); return 2;
        }
        equal(selected_weights[row*2+slot],row<live ? reference_scores[e]/denominator*2.5f : -17.0f,"unbiased selected weight");
      }
    }
    ++cases;
  }
  check(cudaFree(weights)); check(cudaFree(indices)); check(cudaFree(choice));
  check(cudaFree(scores)); check(cudaFree(logits)); check(cudaFree(bias));
  check(cudaFree(weight)); check(cudaFree(bf_weight)); check(cudaFree(input)); check(cudaFree(counts));
  std::printf("routing cases=%d passed; live rows 0/1/2, BF16/F32, sigmoid/softplus, bias and inactive rows\n",cases);
}
