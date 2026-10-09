#include "packed_expert_kernels.cuh"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <vector>
#include <stdexcept>
#include <cmath>
#include <algorithm>

// Independent CPU numeric oracle for the five-stage compact expert program.
// No model weights are loaded: this bounded test validates executable behavior,
// not full-model serving, tensor-core performance, or reference bitwise parity.
static void check(cudaError_t e) {
  if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
template<class T> class Buffer {
  T *p_=nullptr;
public:
  explicit Buffer(size_t n) { check(cudaMalloc(&p_,n*sizeof(T))); }
  ~Buffer() { if(p_) cudaFree(p_); }
  Buffer(const Buffer&)=delete;
  Buffer& operator=(const Buffer&)=delete;
  T *get() { return p_; }
};
static float bf(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }
static float e2(unsigned c) {
  const float v[]={0,.5,1,1.5,2,3,4,6};
  return (c&8)?-v[c&7]:v[c&7];
}
static float e4(unsigned c) {
  unsigned m=c&7,e=(c>>3)&15;
  return e==0?std::ldexp(float(m),-9):std::ldexp(1.f+float(m)/8,int(e)-7);
}
static unsigned encode4(float x) {
  const float v=std::min(448.f,std::fabs(x));
  unsigned best=0; float distance=v;
  for(unsigned c=1;c<=126;++c) {
    float d=std::fabs(v-e4(c));
    if(d<distance||(d==distance&&!(c&1))) { best=c;distance=d; }
  }
  return best|(std::signbit(x)?128:0);
}
static std::vector<float> quantize(const std::vector<float>& input,
                                 std::vector<unsigned char> *packed=nullptr) {
  if(packed) packed->resize(input.size()+input.size()/128);
  std::vector<float> result(input.size());
  for(size_t row=0;row<input.size()/128;++row) {
    float a=0.0001f;
    for(int k=0;k<128;++k) a=std::max(a,std::fabs(input[row*128+k]));
    const float requested=a*(1.f/448.f);
    unsigned scale=0;
    while(scale<254&&std::ldexp(1.f,int(scale)-127)<requested) ++scale;
    if(packed) (*packed)[input.size()+row]=(unsigned char)scale;
    for(int k=0;k<128;++k) {
      unsigned code=encode4(input[row*128+k]/std::ldexp(1.f,int(scale)-127));
      result[row*128+k]=bf((code&128?-1.f:1.f)*e4(code&127)*std::ldexp(1.f,int(scale)-127));
      if(packed) (*packed)[row*128+k]=(unsigned char)code;
    }
  }
  return result;
}
static float dot(const std::vector<float>& a,const std::vector<float>& w,int row) {
  // Reference-style blockwise accumulation, deliberately not the renderer's
  // warp tree. A declared tolerance covers non-associative F32 addition.
  float sum=0;
  for(int block=0;block<4;++block) {
    float partial=0;
    for(int k=block*32;k<(block+1)*32;++k) partial+=a[k]*w[row*128+k];
    sum+=partial;
  }
  return bf(sum);
}
static void run() {
  const int ids[]={3,1},routing[]={3,1,0,3},mapping[]={-1,1,-1,0};
  const float scores[]={.37f,.63f,.5f,.0f};
  std::vector<unsigned char> bank(ds_stride*2,0);
  std::vector<float> weights[2][3];
  for(int owner=0;owner<2;++owner) for(int p=0;p<3;++p) {
    weights[owner][p].resize(128*128);
    auto *target=bank.data()+owner*ds_stride+ds_offsets[p];
    for(int r=0;r<128;++r) for(int c=0;c<128;++c) {
      unsigned code=(ids[owner]*5+p*7+r*3+c)%16;
      unsigned scale=120+(r+p+c/32+owner)%3;
      weights[owner][p][r*128+c]=bf(e2(code)*std::ldexp(1.f,int(scale)-127));
      if(c%2==0) target[r*64+c/2]=(unsigned char)code;
      else target[r*64+c/2]|=(unsigned char)(code<<4);
      target[8192+r*4+c/32]=(unsigned char)scale;
    }
  }
  Buffer<unsigned char> dbank(bank.size()),qx(ds_quantized_bytes[0]),qa(ds_quantized_bytes[1]);
  Buffer<__nv_bfloat16> input(256),activation(512),weighted(512);
  Buffer<int> counts(5),indices(4),map(4),error(1);
  Buffer<float> routing_weights(4),output(256);
  check(cudaMemcpy(dbank.get(),bank.data(),bank.size(),cudaMemcpyHostToDevice));
  check(cudaMemcpy(indices.get(),routing,sizeof(routing),cudaMemcpyHostToDevice));
  check(cudaMemcpy(map.get(),mapping,sizeof(mapping),cudaMemcpyHostToDevice));
  check(cudaMemcpy(routing_weights.get(),scores,sizeof(scores),cudaMemcpyHostToDevice));
  for(bool zero : {false,true}) {
    std::vector<float> x(256);
    std::vector<__nv_bfloat16> xb(256);
    for(int i=0;i<256;++i) {
      x[i]=zero?0.f:bf(float((i*13)%47-23)/16);
      xb[i]=__float2bfloat16_rn(x[i]);
    }
    std::vector<unsigned char> qx_reference;
    const auto rounded_input=quantize(x,&qx_reference);
    for(int live : {2,1,0}) {
      int counter[]={0,0,0,live,0};
      check(cudaMemcpy(counts.get(),counter,sizeof(counter),cudaMemcpyHostToDevice));
      check(cudaMemcpy(input.get(),xb.data(),xb.size()*2,cudaMemcpyHostToDevice));
      check(cudaMemset(error.get(),0,4));
      check(cudaMemset(qx.get(),0xa5,ds_quantized_bytes[0]));
      check(cudaMemset(qa.get(),0xa5,ds_quantized_bytes[1]));
      std::vector<float> got(256,-123.f),expected(256,-123.f);
      check(cudaMemcpy(output.get(),got.data(),got.size()*4,cudaMemcpyHostToDevice));
      ds_quantize_0<<<2,128>>>(counts.get(),input.get(),qx.get(),error.get());
      ds_gate_up<<<128,128>>>(counts.get(),qx.get(),indices.get(),map.get(),dbank.get(),activation.get(),routing_weights.get());
      ds_quantize_1<<<4,128>>>(counts.get(),activation.get(),qa.get(),error.get());
      ds_down<<<128,128>>>(counts.get(),qa.get(),indices.get(),routing_weights.get(),map.get(),dbank.get(),weighted.get());
      ds_combine<<<1,256>>>(counts.get(),indices.get(),weighted.get(),output.get());
      check(cudaGetLastError());check(cudaDeviceSynchronize());
      check(cudaMemcpy(got.data(),output.get(),got.size()*4,cudaMemcpyDeviceToHost));
      int status;check(cudaMemcpy(&status,error.get(),4,cudaMemcpyDeviceToHost));
      if(status) throw std::runtime_error("quantizer reported nonfinite input or scale");
      std::vector<unsigned char> qx_got(ds_quantized_bytes[0]);
      check(cudaMemcpy(qx_got.data(),qx.get(),qx_got.size(),cudaMemcpyDeviceToHost));
      for(int row=0;row<2;++row) {
        for(int c=0;c<128;++c) {
          unsigned char want=row<live?qx_reference[row*128+c]:0xa5;
          if(qx_got[row*128+c]!=want) throw std::runtime_error("FP8 input RNE mismatch");
        }
        unsigned char want=row<live?qx_reference[256+row]:0xa5;
        if(qx_got[256+row]!=want) throw std::runtime_error("UE8M0 amax-floor mismatch");
      }
      for(int row=0;row<live;++row) {
        std::vector<float> in(rounded_input.begin()+row*128,rounded_input.begin()+(row+1)*128);
        for(int c=0;c<128;++c) expected[row*128+c]=0;
        for(int expert : {1,3}) for(int slot=0;slot<2;++slot) {
          if(routing[row*2+slot]!=expert) continue;
          int owner=mapping[expert];std::vector<float> act(128);
          for(int c=0;c<128;++c) {
            float g=std::min(10.f,dot(in,weights[owner][0],c));
            float u=std::max(-10.f,std::min(10.f,dot(in,weights[owner][1],c)));
            // The routing score must precede BF16 rounding and quantization.
            act[c]=bf((g/(1.f+std::exp(-g))*u)*scores[row*2+slot]);
          }
          act=quantize(act);
          for(int c=0;c<128;++c) expected[row*128+c]+=dot(act,weights[owner][2],c);
        }
      }
      float maximum_error=0;
      for(int i=0;i<256;++i) {
        float difference=std::fabs(got[i]-expected[i]);
        maximum_error=std::max(maximum_error,difference);
        if(!std::isfinite(got[i])||difference>0.001f+0.02f*std::fabs(expected[i])) {
          std::fprintf(stderr,"zero=%d live=%d cell=%d got=%g expected=%g\n",zero,live,i,got[i],expected[i]);
          throw std::runtime_error("dynamic expert numerical mismatch");
        }
      }
      std::printf("dynamic-expert zero=%d live=%d max_absolute_error=%g outcome=passed\n",zero,live,maximum_error);
    }
  }
}
int main() {
  try {
    run();check(cudaDeviceReset());
    std::puts("outcome=passed scope=dynamic-fp8-packed-expert-not-full-model");return 0;
  } catch(const std::exception &e) {
    std::fprintf(stderr,"failure=%s\n",e.what());return 1;
  }
}
