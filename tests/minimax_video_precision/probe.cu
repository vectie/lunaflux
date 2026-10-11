// Independent operator-boundary checks on the actual generated FP16 source.
// This executable is test-only; it uses no checkpoint or production library.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "main.cuh"
static void check(cudaError_t e) {
  if (e != cudaSuccess) { std::fprintf(stderr, "%s\n", cudaGetErrorString(e)); std::exit(2); }
}
template<class T> T* allocate(size_t n) {
  T *p=nullptr; check(cudaMallocManaged(&p,n*sizeof(T))); return p;
}
static float rounded(float v) { return __half2float(__float2half_rn(v)); }
static float add(float a,float b) { volatile float v=a+b; return v; }
static float mul(float a,float b) { volatile float v=a*b; return v; }
static void same(float actual,float expected,const char* label,int index,float tolerance=0) {
  if (!std::isfinite(actual) || std::fabs(actual-expected)>tolerance) {
    std::fprintf(stderr,"%s index=%d actual=%g expected=%g\n",label,index,actual,expected); std::exit(3);
  }
}
int main() {
  auto input=allocate<float>(132),weight=allocate<float>(576),bias=allocate<float>(48);
  auto registers=allocate<float>(48),front=allocate<float>(18),embedded=allocate<float>(132);
  auto low=allocate<__half>(396),qkv=allocate<__half>(396),rope=allocate<__half>(66);
  auto normalized=allocate<__half>(396),attention=allocate<__half>(132),product=allocate<__half>(264);
  auto delta=allocate<__half>(132);
  auto residual=allocate<float>(132),scale=allocate<float>(12);
  for(int i=0;i<132;++i)input[i]=float((i*7)%19-9)/16;
  for(int i=0;i<576;++i)weight[i]=float((i*3)%11-5)/32+1.0f/4096;
  for(int i=0;i<48;++i)bias[i]=float(i%7-3)/64+1.0f/4096;
  for(int i=0;i<48;++i)registers[i]=float(i)/128;
  post_quant<<<1,256>>>(input,weight,bias,front);
  check(cudaGetLastError()); check(cudaDeviceSynchronize());
  for(int c=0;c<3;++c)for(int t=0;t<6;++t) {
    float sum=0;for(int k=0;k<3;++k)sum=add(sum,mul(rounded(input[k*6+t]),rounded(weight[c*3+k])));
    same(front[c*6+t],rounded(add(sum,rounded(bias[c]))),"post-quant",c*6+t);
  }
  embed<<<1,256>>>(front,weight,bias,registers,embedded);
  check(cudaGetLastError());check(cudaDeviceSynchronize());
  for(int r=0;r<11;++r)for(int c=0;c<12;++c) {
    float expected=0;
    if(r<6) { for(int k=0;k<3;++k)expected=add(expected,mul(front[k*6+r],weight[c*3+k])); expected=rounded(add(expected,bias[c])); }
    else if(r<10)expected=registers[(r-6)*12+c];
    same(embedded[r*12+c],expected,"embedding",r*12+c);
  }
  for(int c=0;c<12;++c)scale[c]=1;
  probe_norm<<<11,1>>>(embedded,scale,low);
  probe_qkv<<<2,256>>>(low,weight,bias,weight,bias,weight,bias,qkv);
  probe_rotary_table<<<1,256>>>(rope);
  probe_qk_rope<<<66,1>>>(qkv,rope,normalized);
  check(cudaGetLastError());check(cudaDeviceSynchronize());
  for(int r=0;r<11;++r) {
    float square=0;for(int c=0;c<12;++c)square=add(square,mul(embedded[r*12+c],embedded[r*12+c]));
    const float inv=1/std::sqrt(add(square/12,1.0e-5f));
    for(int c=0;c<12;++c)same(__half2float(low[r*12+c]),rounded(mul(embedded[r*12+c],inv)),"norm",r*12+c,.001f);
    for(int p=0;p<3;++p)for(int c=0;c<12;++c) {
      float sum=0;for(int k=0;k<12;++k)sum=add(sum,mul(__half2float(low[r*12+k]),rounded(weight[c*12+k])));
      same(__half2float(qkv[p*132+r*12+c]),rounded(add(sum,rounded(bias[c]))),"qkv",p*132+r*12+c);
    }
  }
  for(int r=0;r<11;++r)for(int axis=0;axis<3;++axis) {
    const int coordinate=axis==0?0:axis==1?r/3:r%3,extent=axis==0?1:axis==1?2:3;
    const float position=r<6?rounded(add(rounded(mul(2,rounded(rounded(add(float(coordinate),.5f))/extent))),-1)):0;
    const float angle=rounded(mul(6.2831853071795864769f,position));
    same(__half2float(rope[r*6+axis]),rounded(std::cos(angle)),"rotary-cos",r*6+axis,.001f);
    same(__half2float(rope[r*6+3+axis]),rounded(std::sin(angle)),"rotary-sin",r*6+3+axis,.001f);
  }
  // The value plane is passed through unchanged, with no Q/K normalization.
  for(int i=264;i<396;++i)same(__half2float(normalized[i]),__half2float(qkv[i]),"value-copy",i);
  probe_attention<<<22,1>>>(normalized,attention);
  probe_project<<<1,256>>>(attention,weight,bias,delta);
  probe_residual<<<1,256>>>(embedded,delta,scale,residual);
  probe_gate_up<<<2,256>>>(low,weight,bias,product);
  probe_down<<<1,256>>>(product,weight,bias,delta);
  check(cudaGetLastError());check(cudaDeviceSynchronize());
  for(int i=0;i<132;++i) {
    if(!std::isfinite(residual[i]) || !std::isfinite(__half2float(delta[i])))return 4;
    float sum=0;const int r=i/12,c=i%12;
    for(int k=0;k<24;++k)sum=add(sum,mul(__half2float(product[r*24+k]),rounded(weight[c*24+k])));
    same(__half2float(delta[i]),rounded(add(sum,rounded(bias[c]))),"down",i);
  }
  void *allocations[]={input,weight,bias,registers,front,embedded,
      low,qkv,rope,normalized,attention,product,delta,residual,scale};
  for(void *p : allocations)check(cudaFree(p));
  check(cudaDeviceReset());
  std::puts("video_fp16_actual_lowering=true frontend_qkv_rotary_down_checks=passed finite_tail=true allocations_released=true");
}
