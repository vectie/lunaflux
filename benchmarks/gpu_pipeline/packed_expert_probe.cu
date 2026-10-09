#include "packed_expert_kernels.cuh"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <vector>
#include <stdexcept>
#include <cmath>

// Small deterministic execution probe, not a serving/performance benchmark.
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
  return (c&8) ? -v[c&7] : v[c&7];
}
static unsigned code(int expert,int projection,int row,int column) {
  // Unequal signs/magnitudes and row/column dependence exercise both nibbles.
  return (unsigned)((expert*5+projection*7+row*3+column)%16);
}
static float dot(const std::vector<float>& a,const std::vector<float>& w,int row) {
  float lane[32];
  for(int k=0;k<32;++k) lane[k]=a[k]*w[row*32+k];
  for(int d=16;d>0;d>>=1)
    for(int k=0;k<32-d;++k) lane[k]=lane[k]+lane[k+d];
  return lane[0];
}
static void run(int format) {
  const char *name=format==0?"NVFP4":format==1?"MXFP4":"BF16";
  const size_t stride=format==0?nv_stride:format==1?mx_stride:bf_stride;
  const size_t *offsets=format==0?nv_offsets:format==1?mx_offsets:bf_offsets;
  std::vector<unsigned char> bank(stride*2,0xa7);
  std::vector<float> w[2][3];
  const int global_ids[]={3,1};
  for(int expert=0;expert<2;++expert) for(int p=0;p<3;++p) {
    w[expert][p].resize(1024);
    auto *target=bank.data()+expert*stride+offsets[p];
    for(int r=0;r<32;++r) for(int c=0;c<32;++c) {
      const unsigned q=code(global_ids[expert],p,r,c);
      const float value=bf(e2(q)*(format==0?.625f:format==1?.5f:.25f));
      w[expert][p][r*32+c]=value;
      if(format==2) {
        const __nv_bfloat16 v=__float2bfloat16_rn(value);
        std::memcpy(target+(r*32+c)*2,&v,2);
      } else {
        if(c%2==0) target[r*16+c/2]=(unsigned char)q;
        else target[r*16+c/2]|=(unsigned char)(q<<4);
      }
    }
    if(format==0) {
      // E4M3 .5 and global F32 1.25, including alignment padding.
      std::memset(target+512,0x30,64);
      const float global=1.25f; std::memcpy(target+576,&global,4);
    } else if(format==1) std::memset(target+512,126,32);
  }
  Buffer<unsigned char> dbank(bank.size());
  Buffer<__nv_bfloat16> input(64),activation(128),weighted(128);
  Buffer<int> counts(5),indices(4),map(4);
  Buffer<float> weights(4),output(64);
  const int route[]={3,1,0,3},mapping[]={-1,1,-1,0};
  const float scores[]={.25f,.75f,.5f,.5f};
  std::vector<__nv_bfloat16> x(64);
  for(int i=0;i<64;++i) x[i]=__float2bfloat16_rn((float)(i%9-4)/16);
  check(cudaMemcpy(dbank.get(),bank.data(),bank.size(),cudaMemcpyHostToDevice));
  check(cudaMemcpy(input.get(),x.data(),128,cudaMemcpyHostToDevice));
  check(cudaMemcpy(indices.get(),route,sizeof(route),cudaMemcpyHostToDevice));
  check(cudaMemcpy(map.get(),mapping,sizeof(mapping),cudaMemcpyHostToDevice));
  check(cudaMemcpy(weights.get(),scores,sizeof(scores),cudaMemcpyHostToDevice));
  for(int live : {2,1,0}) {
    const int counter[]={0,0,0,live,0};
    std::vector<float> got(64,-123.0f);
    check(cudaMemcpy(counts.get(),counter,sizeof(counter),cudaMemcpyHostToDevice));
    check(cudaMemcpy(output.get(),got.data(),256,cudaMemcpyHostToDevice));
    check(cudaMemset(activation.get(),0xcd,256));
    check(cudaMemset(weighted.get(),0xcd,256));
    if(format==0) {
      nv_gate_up<<<32,128>>>(counts.get(),input.get(),indices.get(),map.get(),dbank.get(),activation.get());
      nv_down<<<32,128>>>(counts.get(),activation.get(),indices.get(),weights.get(),map.get(),dbank.get(),weighted.get());
      nv_combine<<<1,256>>>(counts.get(),indices.get(),weighted.get(),output.get());
    } else if(format==1) {
      mx_gate_up<<<32,128>>>(counts.get(),input.get(),indices.get(),map.get(),dbank.get(),activation.get());
      mx_down<<<32,128>>>(counts.get(),activation.get(),indices.get(),weights.get(),map.get(),dbank.get(),weighted.get());
      mx_combine<<<1,256>>>(counts.get(),indices.get(),weighted.get(),output.get());
    } else {
      bf_gate_up<<<32,128>>>(counts.get(),input.get(),indices.get(),map.get(),dbank.get(),activation.get());
      bf_down<<<32,128>>>(counts.get(),activation.get(),indices.get(),weights.get(),map.get(),dbank.get(),weighted.get());
      bf_combine<<<1,256>>>(counts.get(),indices.get(),weighted.get(),output.get());
    }
    check(cudaGetLastError()); check(cudaDeviceSynchronize());
    check(cudaMemcpy(got.data(),output.get(),256,cudaMemcpyDeviceToHost));
    for(int row=0;row<2;++row) {
      std::vector<float> expected(32,0),in(32);
      for(int c=0;c<32;++c) in[c]=__bfloat162float(x[row*32+c]);
      // Canonical global expert order, regardless of routing slot order.
      for(int expert : {1,3}) for(int slot=0;slot<2;++slot) {
        if(route[row*2+slot]!=expert) continue;
        const int owner=mapping[expert]; std::vector<float> act(32);
        for(int c=0;c<32;++c) {
          const float g=fminf(bf(dot(in,w[owner][0],c)),10.0f);
          const float u=fmaxf(-10.0f,fminf(bf(dot(in,w[owner][1],c)),10.0f));
          act[c]=bf(bf(g/(1+expf(-g)))*u);
        }
        for(int c=0;c<32;++c)
          expected[c]+=bf(bf(dot(act,w[owner][2],c))*scores[row*2+slot]);
      }
      for(int c=0;c<32;++c) {
        const float want=row<live?expected[c]:-123.0f;
        if(!std::isfinite(got[row*32+c]) || fabsf(got[row*32+c]-want)>0.002f) {
          std::fprintf(stderr,"format=%s live=%d row=%d col=%d got=%g expected=%g\n",name,live,row,c,got[row*32+c],want);
          throw std::runtime_error("packed expert numerical mismatch");
        }
      }
    }
  }
  std::printf("format=%s outcome=passed live_rows=2,1,0 local_experts=3,1\n",name);
}
int main() {
  try {
    for(int format=0;format<3;++format) run(format);
    check(cudaDeviceReset());
    std::puts("outcome=passed scope=compact-expert-compute-not-full-model");
    return 0;
  } catch(const std::exception &e) {
    std::fprintf(stderr,"failure=%s\n",e.what()); return 1;
  }
}
