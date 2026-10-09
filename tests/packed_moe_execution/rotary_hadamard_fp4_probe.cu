#include "rotary_hadamard_fp4_kernels.cuh"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CK(expr) do { const cudaError_t e = (expr); if(e != cudaSuccess) { \
  std::fprintf(stderr,"%s: %s\n",#expr,cudaGetErrorString(e)); std::exit(2); \
} } while(0)

template<class T> T *upload(const std::vector<T>& data) {
  T *p; CK(cudaMalloc(&p,data.size()*sizeof(T)));
  CK(cudaMemcpy(p,data.data(),data.size()*sizeof(T),cudaMemcpyHostToDevice)); return p;
}
template<class T> void read(std::vector<T>& data,T *p) {
  CK(cudaMemcpy(data.data(),p,data.size()*sizeof(T),cudaMemcpyDeviceToHost));
}
float f(__nv_bfloat16 x) { return __bfloat162float(x); }
__nv_bfloat16 b(float x) { return __float2bfloat16_rn(x); }
float scale_for(float maximum) {
  const float requested = std::max(maximum, std::ldexp(6.0f,-126)) * (1.0f/6.0f);
  uint32_t bits; std::memcpy(&bits,&requested,4);
  bits = (((bits >> 23) & 255U) + ((bits & 0x7fffffU) != 0U)) << 23;
  float scale; std::memcpy(&scale,&bits,4); return scale;
}
float quantize(float value,float scale) {
  constexpr float levels[8] = {0,0.5f,1,1.5f,2,3,4,6};
  const float normalized = std::min(6.0f,std::fabs(value/scale));
  int selected = 0; float distance = normalized;
  for(int code=1;code<8;++code) {
    const float delta = std::fabs(normalized-levels[code]);
    if(delta < distance || (delta == distance && !(code & 1))) { selected=code; distance=delta; }
  }
  return std::copysign(levels[selected]*scale,value);
}

int main() {
  constexpr int rows=3,heads=2,width=128,suffix=64,block=32,size=rows*heads*width;
  std::vector<__nv_bfloat16> input(size),rotated(size,b(71)),transformed(size,b(73));
  std::vector<int32_t> counts(5),positions{0,3,1023};
  for(int i=0;i<size;++i) input[i]=b(float((i*7)%29-14)/32);
  auto x=upload(input), y=upload(rotated), z=upload(transformed);
  auto c=upload(counts), p=upload(positions);
  float largest=0;
  for(int live : {0,1,2,3,1,3}) {
    counts[2]=live?1:0; counts[3]=live;
    CK(cudaMemcpy(c,counts.data(),20,cudaMemcpyHostToDevice));
    const auto inactive=transformed;
    hadamard_fp4_rotary<<<dim3(rows,heads),256>>>(c,p,(uint16_t*)x,(uint16_t*)y);
    hadamard_fp4_hadamard<<<dim3(rows,heads),256>>>(c,y,z);
    hadamard_fp4_simulate<<<dim3(rows,heads*(width/block)),block>>>(c,z);
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize()); read(rotated,y); read(transformed,z);
    for(int row=0;row<rows;++row) for(int head=0;head<heads;++head) {
      const int base=(row*heads+head)*width;
      if(row>=live) {
        for(int col=0;col<width;++col) if(f(transformed[base+col])!=f(inactive[base+col])) {
          std::fprintf(stderr,"inactive row changed\n"); return 3;
        }
        continue;
      }
      std::vector<float> values(width);
      for(int col=0;col<width;++col) values[col]=f(input[base+col]);
      for(int pair=0;pair<suffix/2;++pair) {
        const float frequency=1.0f/std::pow(10000.0f,float(pair*2)/suffix);
        const float angle=float(positions[row])*frequency;
        const int index=width-suffix+2*pair;
        const float left=values[index],right=values[index+1];
        values[index]=f(b(left*std::cos(angle)-right*std::sin(angle)));
        values[index+1]=f(b(left*std::sin(angle)+right*std::cos(angle)));
      }
      for(int col=0;col<width;++col) if(std::fabs(f(rotated[base+col])-values[col])>0.0078125f) return 4;
      for(int stride=1;stride<width;stride*=2) for(int start=0;start<width;start+=2*stride)
        for(int col=0;col<stride;++col) {
          const float left=values[start+col],right=values[start+stride+col];
          values[start+col]=left+right; values[start+stride+col]=left-right;
        }
      const float normalization = float(1.0/std::sqrt(double(width)));
      for(float& value:values) value=f(b(value*normalization));
      for(int start=0;start<width;start+=block) {
        float maximum=0; for(int col=0;col<block;++col) maximum=std::max(maximum,std::fabs(values[start+col]));
        const float scale=scale_for(maximum);
        for(int col=0;col<block;++col) {
          const float expected=f(b(quantize(values[start+col],scale)));
          const float error=std::fabs(f(transformed[base+start+col])-expected); largest=std::max(largest,error);
          if(error>0.015625f+0.01f*std::fabs(expected)) { std::fprintf(stderr,"Hadamard FP4 mismatch live=%d row=%d head=%d col=%d expected=%g actual=%g before=%g scale=%g error=%g\n",live,row,head,start+col,expected,f(transformed[base+start+col]),values[start+col],scale,error); return 5; }
        }
      }
    }
  }
  // Test E2M1 ties directly at scale one, including both signs and signed zero.
  constexpr float boundary[16]={0,0.25f,0.75f,1.25f,1.75f,2.5f,3.5f,5,6,-0.25f,-0.75f,-1.25f,-1.75f,-2.5f,-3.5f,-5};
  for(int i=0;i<size;++i) transformed[i]=b(boundary[i%16]);
  CK(cudaMemcpy(z,transformed.data(),size*2,cudaMemcpyHostToDevice));
  counts[3]=3; CK(cudaMemcpy(c,counts.data(),20,cudaMemcpyHostToDevice));
  hadamard_fp4_simulate<<<dim3(rows,heads*(width/block)),block>>>(c,z); CK(cudaDeviceSynchronize()); read(transformed,z);
  for(int i=0;i<size;++i) if(f(transformed[i])!=quantize(boundary[i%16],1.0f)) return 6;
  for(void *ptr : {(void*)x,(void*)y,(void*)z,(void*)c,(void*)p}) CK(cudaFree(ptr));
  std::printf("rotary_hadamard_fp4 live=0/1/2/3 replay=passed midpoint_ties=passed largest_error=%.9g\n",largest);
  return 0;
}
