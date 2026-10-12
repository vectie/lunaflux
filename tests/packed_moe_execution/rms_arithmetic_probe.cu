// Offline matched RMS arithmetic. The vector fold is an explicit summation
// order, not an approximation or a replacement reference. No production ABI.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

static void check(cudaError_t e) {
  if(e!=cudaSuccess){std::fprintf(stderr,"%s\n",cudaGetErrorString(e));std::exit(2);}
}
template<class T> static T *load(const std::string &path,size_t n) {
  auto *file=std::fopen(path.c_str(),"rb");if(!file){std::perror(path.c_str());std::exit(3);}
  std::vector<T> values(n);const bool read=std::fread(values.data(),sizeof(T),n,file)==n&&std::fgetc(file)==EOF;
  std::fclose(file);if(!read)std::exit(3);
  T *p=nullptr;check(cudaMalloc(&p,n*sizeof(T)));check(cudaMemcpy(p,values.data(),n*sizeof(T),cudaMemcpyHostToDevice));return p;
}
template<class T> static T *allocate(size_t n) {
  T *p=nullptr;check(cudaMalloc(&p,n*sizeof(T)));check(cudaMemset(p,0,n*sizeof(T)));return p;
}
template<class T> static void save(T *p,size_t n,const std::string &path) {
  std::vector<T> values(n);check(cudaMemcpy(values.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost));
  auto *file=std::fopen(path.c_str(),"wbx");if(!file){std::perror(path.c_str());std::exit(4);}
  const bool wrote=std::fwrite(values.data(),sizeof(T),n,file)==n,closed=std::fclose(file)==0;
  if(!wrote||!closed)std::exit(4);
}

template<int Width,int Partitions,int VectorWidth,bool LiteralMean,bool ReciprocalSqrt>
__global__ void probe(const __nv_bfloat16 *x,const __nv_bfloat16 *weights,
    const float *literal_mean,float *means,float *inverse,__nv_bfloat16 *normalized) {
  const int row=blockIdx.x,lane=threadIdx.x;
  __shared__ float partial[Partitions];
  float sum[VectorWidth]={};
  for(int base=lane*VectorWidth;base<Width;base+=Partitions*VectorWidth) {
#pragma unroll
    for(int component=0;component<VectorWidth;++component) {
      const float value=__bfloat162float(x[row*Width+base+component]);
      sum[component]=__fadd_rn(sum[component],__fmul_rn(value,value));
    }
  }
#pragma unroll
  for(int component=1;component<VectorWidth;++component)sum[0]=__fadd_rn(sum[0],sum[component]);
  partial[lane]=sum[0];__syncthreads();
  for(int span=Partitions/2;span>0;span>>=1) {
    if(lane<span)partial[lane]=__fadd_rn(partial[lane],partial[lane+span]);
    __syncthreads();
  }
  const float mean=LiteralMean?literal_mean[row]:__fdiv_rn(partial[0],float(Width));
  const float argument=__fadd_rn(mean,0.000001f);
  const float inv=ReciprocalSqrt?rsqrtf(argument):__fdiv_rn(1.0f,sqrtf(argument));
  if(lane==0){means[row]=mean;inverse[row]=inv;}
  if(normalized)for(int column=lane;column<Width;column+=Partitions) {
    const float value=__bfloat162float(x[row*Width+column]);
    const float scale=__bfloat162float(weights[column]);
    normalized[row*Width+column]=__float2bfloat16_rn(__fmul_rn(__fmul_rn(value,inv),scale));
  }
}

int main(int argc,char **argv) {
  if(argc!=3)return 2;
  const std::string input(argv[1]),out(argv[2]);
  auto *streams=load<__nv_bfloat16>(input+"/streams.bf16",144*16384);
  auto *collapsed=load<__nv_bfloat16>(input+"/collapsed.bf16",144*4096);
  auto *weight=load<__nv_bfloat16>(input+"/norm.bf16",4096);
  auto *control_mean=load<float>(input+"/control-mean-replay.f32",144);
  auto *norm_mean=load<float>(input+"/norm-variance.f32",144);
  auto *means=allocate<float>(144),*inverse=allocate<float>(144);
  auto *normalized=allocate<__nv_bfloat16>(144*4096);
#define RUN(P,V,L,R,label) do { \
  probe<16384,P,V,L,R><<<144,P>>>(streams,nullptr,control_mean,means,inverse,nullptr); \
  check(cudaGetLastError());check(cudaDeviceSynchronize()); \
  save(means,144,out+"/" label "-control-mean.f32");save(inverse,144,out+"/" label "-control-rsqrt.f32"); \
  probe<4096,P,V,L,R><<<144,P>>>(collapsed,weight,norm_mean,means,inverse,normalized); \
  check(cudaGetLastError());check(cudaDeviceSynchronize()); \
  save(means,144,out+"/" label "-norm-variance.f32");save(inverse,144,out+"/" label "-norm-rsqrt.f32"); \
  save(normalized,144*4096,out+"/" label "-normalized.bf16"); \
} while(false)
  RUN(256,1,false,true,"stripe256-rsqrt");
  RUN(512,1,false,true,"stripe512-rsqrt");
  RUN(256,4,false,true,"vector256-rsqrt");
  RUN(512,4,false,true,"vector512-rsqrt");
  RUN(512,4,false,false,"vector512-divsqrt");
  RUN(512,4,true,true,"literal-mean-rsqrt");
  RUN(512,4,true,false,"literal-mean-divsqrt");
  for(void *p:{(void*)streams,(void*)collapsed,(void*)weight,(void*)control_mean,(void*)norm_mean,
    (void*)means,(void*)inverse,(void*)normalized})check(cudaFree(p));
  std::puts("rms_arithmetic_probe rows=144 control_width=16384 norm_width=4096 offline_only=true reference_math=unchanged release=balanced");
}
