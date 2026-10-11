// Independent F64 oracle on the exact generated projection/MLP source.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "main.cuh"

static void check(cudaError_t e) {
  if(e != cudaSuccess) { std::fprintf(stderr,"%s\n",cudaGetErrorString(e)); std::exit(2); }
}
template<class T> static T *allocate(size_t n) {
  T *p=nullptr; check(cudaMallocManaged(&p,n*sizeof(T))); return p;
}
static __nv_bfloat16 bf(double x) { return __float2bfloat16_rn(float(x)); }
static unsigned bits(__nv_bfloat16 x) { unsigned short n; std::memcpy(&n,&x,2); return n; }
int main() {
  auto counts=allocate<int>(4); counts[3]=1;
  auto x=allocate<__nv_bfloat16>(5120),w=allocate<__nv_bfloat16>(4*5120);
  auto y=allocate<__nv_bfloat16>(4),old=allocate<__nv_bfloat16>(4);
  for(int i=0;i<5120;++i) x[i]=bf(1);
  for(int row=0;row<4;++row) for(int i=0;i<5120;++i) {
    double v=0;
    if(row<2) v=i==0?(row? -16777216:16777216):i==5119?(row?16777216:-16777216):(row?-1:1);
    else if(row==2) v=double((i*17)%101-50)/32;
    else v=i%2? -double((i/2)%71)/16:double((i/2)%71)/16;
    w[row*5120+i]=bf(v);
  }
  std::vector<__nv_bfloat16> expected(4);
  for(int row=0;row<4;++row) {
    double sum=0; for(int i=0;i<5120;++i)
      sum+=double(__bfloat162float(x[i]))*double(__bfloat162float(w[row*5120+i]));
    expected[row]=bf(sum);
  }
  int recovered=0;
  for(int repeat=0;repeat<2;++repeat) {
    compensated_dot<<<1,256>>>(counts,x,w,y);
    ordered_dot<<<1,256>>>(counts,x,w,old);
    check(cudaGetLastError()); check(cudaDeviceSynchronize());
    for(int row=0;row<4;++row) {
      if(bits(y[row])!=bits(expected[row])) {
        std::fprintf(stderr,"dot row=%d actual=%g F64=%g\n",row,__bfloat162float(y[row]),__bfloat162float(expected[row])); return 3;
      }
      if(repeat==0 && bits(old[row])!=bits(expected[row])) ++recovered;
    }
  }
  if(recovered!=2) return 4;
  // Exercise both MLP stages with cancellation in the up dot, preserving
  // explicit BF16 boundaries after projection, SiLU and product.
  for(int i=0;i<8;++i) x[i]=bf(1);
  for(int i=0;i<48;++i) w[i]=bf(0);
  for(int c=0;c<3;++c) {
    w[c*8]=bf(16777216); w[c*8+1]=bf(1); w[c*8+2]=bf(-16777216);
    w[(c+3)*8+1]=bf(1);
  }
  auto product=allocate<__nv_bfloat16>(3),down=allocate<__nv_bfloat16>(24),out=allocate<__nv_bfloat16>(8);
  for(int i=0;i<24;++i) down[i]=bf(1);
  compensated_mlp_gate_up<<<1,256>>>(x,w,product);
  compensated_mlp_down<<<1,256>>>(product,down,out);
  check(cudaGetLastError()); check(cudaDeviceSynchronize());
  const auto activation=bf(1.0/(1.0+std::exp(-1.0)));
  const auto result=bf(3.0*double(__bfloat162float(activation)));
  for(int i=0;i<3;++i) if(bits(product[i])!=bits(activation)) return 5;
  for(int i=0;i<8;++i) if(bits(out[i])!=bits(result)) return 6;
  check(cudaFree(out)); check(cudaFree(down)); check(cudaFree(product));
  check(cudaFree(old)); check(cudaFree(y)); check(cudaFree(w)); check(cudaFree(x)); check(cudaFree(counts));
  check(cudaDeviceReset());
  std::printf("compensated_dot_F64_exact=8/8 ordered_cancellation_errors_recovered=2/4 mlp_boundaries_exact=true allocations_released=true\n");
}
