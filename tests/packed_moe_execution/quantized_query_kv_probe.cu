// Independent blockwise CPU oracle for six projection/normalization boundaries.
// Small numerical fixture, not full-model or serving performance evidence.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "query_kv_kernels.cuh"

static void check(cudaError_t error) {
  if (error != cudaSuccess) { std::fprintf(stderr, "%s\n", cudaGetErrorString(error)); std::exit(2); }
}
template<class T> static T *allocate(size_t count) {
  T *result = nullptr;
  check(cudaMallocManaged(&result, count * sizeof(T)));
  std::memset(result, 0, count * sizeof(T));
  return result;
}
static float bf(float value) { return __bfloat162float(__float2bfloat16_rn(value)); }
static float e4(unsigned char bits) { __nv_fp8_e4m3 value; value.__x=bits; return (float)value; }
static float rounded_e4(float value) {
  value=std::max(-448.f,std::min(448.f,value));
  if (value==0.f) value=0.f;
  return e4(__nv_cvt_float_to_fp8(value,__NV_SATFINITE,__NV_E4M3));
}
static void projection(const float *input, const unsigned char *weight,
                       const unsigned char *scales, float *output,
                       int live, int in, int out) {
  for (int row=0;row<live;++row) for (int column=0;column<out;++column) {
    float sum=0.f;
    for (int block=0;block<in/128;++block) {
      float maximum=0.0001f;
      for (int i=0;i<128;++i) maximum=std::max(maximum,std::fabs(input[row*in+block*128+i]));
      const float requested=maximum/448.f;
      int exponent=0;
      const float mantissa=std::frexp(requested,&exponent);
      const float scale=std::ldexp(1.f,mantissa==0.5f ? exponent-1 : exponent);
      const float parameter_scale=std::ldexp(1.f,int(scales[(column/128)*(in/128)+block])-127);
      float block_sum=0.f;
      for (int i=0;i<128;++i) {
        const float activation=rounded_e4(input[row*in+block*128+i]/scale);
        block_sum=block_sum+activation*e4(weight[column*in+block*128+i]);
      }
      sum=sum+block_sum*(scale*parameter_scale);
    }
    output[row*out+column]=bf(sum);
  }
}
static void learned_norm(const float *input, const __nv_bfloat16 *weight,
                         float *output, int live, int width) {
  for (int row=0;row<live;++row) {
    float sum=0.f;
    for (int c=0;c<width;++c) sum=sum+input[row*width+c]*input[row*width+c];
    const float inverse=1.f/std::sqrt(sum/float(width)+1.e-6f);
    for (int c=0;c<width;++c) output[row*width+c]=bf((input[row*width+c]*inverse)*__bfloat162float(weight[c]));
  }
}
static void head_norm(const float *input, float *output, int live, int heads, int width) {
  for (int row=0;row<live;++row) for (int head=0;head<heads;++head) {
    const int base=(row*heads+head)*width;
    float sum=0.f;
    for (int c=0;c<width;++c) sum=sum+bf(input[base+c]*input[base+c]);
    const float mean=bf(sum/float(width)), variance=bf(mean+1.e-6f);
    const float inverse=bf(1.f/std::sqrt(variance));
    for (int c=0;c<width;++c) output[base+c]=bf(input[base+c]*inverse);
  }
}
int main() {
  constexpr int R=3,H=256,Q=128,N=2,D=128;
  auto counts=allocate<int>(5);
  auto input=allocate<__nv_bfloat16>(R*H);
  auto qa=allocate<unsigned char>(Q*H), qb=allocate<unsigned char>(N*D*Q), kv=allocate<unsigned char>(D*H);
  auto sa=allocate<unsigned char>(2), sb=allocate<unsigned char>(2), sk=allocate<unsigned char>(2);
  auto qnorm=allocate<__nv_bfloat16>(Q), knorm=allocate<__nv_bfloat16>(D);
  const int widths[6]={Q,Q,N*D,N*D,D,D};
  __nv_bfloat16 *buffers[6];
  std::vector<float> expected[6];
  for (int s=0;s<6;++s) { buffers[s]=allocate<__nv_bfloat16>(R*widths[s]); expected[s].resize(R*widths[s]); }
  for (int i=0;i<Q*H;++i) qa[i]=(unsigned char)(16+(i*7)%32+((i%3)==0?128:0));
  for (int i=0;i<N*D*Q;++i) qb[i]=(unsigned char)(16+(i*3)%32+((i%5)==0?128:0));
  for (int i=0;i<D*H;++i) kv[i]=(unsigned char)(16+(i*11)%32+((i%7)==0?128:0));
  for (int i=0;i<2;++i) { sa[i]=124+i; sb[i]=125+i; sk[i]=126-i; }
  for (int i=0;i<Q;++i) qnorm[i]=__float2bfloat16_rn(0.75f+float(i%9)*0.03125f);
  for (int i=0;i<D;++i) knorm[i]=__float2bfloat16_rn(0.5f+float(i%7)*0.0625f);
  auto run=[&]() {
    query_fixture_q_a<<<dim3(R,1),256>>>(counts,input,qa,sa,buffers[0]);
    query_fixture_q_norm<<<R,256>>>(counts,buffers[0],qnorm,buffers[1]);
    query_fixture_q_b<<<dim3(R,1),256>>>(counts,buffers[1],qb,sb,buffers[2]);
    query_fixture_q_head_norm<<<dim3(R,N),256>>>(counts,buffers[2],buffers[3]);
    query_fixture_kv<<<dim3(R,1),256>>>(counts,input,kv,sk,buffers[4]);
    query_fixture_kv_norm<<<R,256>>>(counts,buffers[4],knorm,buffers[5]);
    check(cudaGetLastError()); check(cudaDeviceSynchronize());
  };
  for (int zeros : {0,1}) for (int live : {0,1,3}) {
    counts[3]=live;
    std::vector<float> hidden(R*H);
    for (int i=0;i<R*H;++i) { hidden[i]=bf(zeros?0.f:float((i*13)%31-15)*0.125f); input[i]=__float2bfloat16_rn(hidden[i]); }
    projection(hidden.data(),qa,sa,expected[0].data(),live,H,Q);
    learned_norm(expected[0].data(),qnorm,expected[1].data(),live,Q);
    projection(expected[1].data(),qb,sb,expected[2].data(),live,Q,N*D);
    head_norm(expected[2].data(),expected[3].data(),live,N,D);
    projection(hidden.data(),kv,sk,expected[4].data(),live,H,D);
    learned_norm(expected[4].data(),knorm,expected[5].data(),live,D);
    for (int s=0;s<6;++s) for (int i=0;i<R*widths[s];++i) buffers[s][i]=__float2bfloat16_rn(19.f);
    std::vector<__nv_bfloat16> first[6];
    for (int repeat=0;repeat<2;++repeat) {
      run();
      for (int s=0;s<6;++s) {
        float maximum=0.f;
        for (int i=0;i<live*widths[s];++i) maximum=std::max(maximum,std::fabs(__bfloat162float(buffers[s][i])-expected[s][i]));
        if (maximum>0.03125f) { std::fprintf(stderr,"stage=%d error=%.9g\n",s,maximum); return 3; }
        for (int i=live*widths[s];i<R*widths[s];++i) if (__bfloat162float(buffers[s][i])!=19.f) return 4;
        if (repeat==0) first[s].assign(buffers[s],buffers[s]+R*widths[s]);
        else if (std::memcmp(first[s].data(),buffers[s],R*widths[s]*sizeof(__nv_bfloat16))) return 5;
        std::printf("zero=%d live=%d repeat=%d stage=%d maxabs=%.9g inactive=untouched\n",zeros,live,repeat,s,maximum);
      }
    }
  }
  for (auto buffer : buffers) check(cudaFree(buffer));
  check(cudaFree(knorm)); check(cudaFree(qnorm)); check(cudaFree(sk)); check(cudaFree(sb)); check(cudaFree(sa));
  check(cudaFree(kv)); check(cudaFree(qb)); check(cudaFree(qa)); check(cudaFree(input)); check(cudaFree(counts));
  std::puts("quantized_query_kv=passed six_stages=true deterministic=true all_allocations=released");
}
