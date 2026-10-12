// Offline exact-operand dot-order isolation. No production dependency or
// literal-reference substitution. Each route computes its own control RMS.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

static void check(cudaError_t e) {
  if(e!=cudaSuccess){std::fprintf(stderr,"%s\n",cudaGetErrorString(e));std::exit(2);}
}
static void blas(cublasStatus_t e) {
  if(e!=CUBLAS_STATUS_SUCCESS){std::fprintf(stderr,"cublas status=%d\n",int(e));std::exit(2);}
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

__global__ void prepare(const __nv_bfloat16 *x,float *wide,float *inverse) {
  const int row=blockIdx.x,lane=threadIdx.x;
  __shared__ float partial[512];float sums[4]={};
  for(int base=lane*4;base<16384;base+=2048) {
#pragma unroll
    for(int v=0;v<4;++v) {
      const float value=__bfloat162float(x[row*16384+base+v]);
      wide[row*16384+base+v]=value;
      sums[v]=__fadd_rn(sums[v],__fmul_rn(value,value));
    }
  }
#pragma unroll
  for(int v=1;v<4;++v)sums[0]=__fadd_rn(sums[0],sums[v]);
  partial[lane]=sums[0];__syncthreads();
  for(int span=256;span>0;span>>=1) {
    if(lane<span)partial[lane]=__fadd_rn(partial[lane],partial[lane+span]);
    __syncthreads();
  }
  if(lane==0)inverse[row]=rsqrtf(__fadd_rn(__fdiv_rn(partial[0],16384.0f),0.000001f));
}

template<int P,int V,bool Fma> __global__ void projection(
    const float *x,const float *weight,const float *inverse,float *mixes) {
  const int row=blockIdx.x,output=blockIdx.y,lane=threadIdx.x;
  __shared__ float partial[P];float sums[V]={};
  for(int base=lane*V;base<16384;base+=P*V) {
#pragma unroll
    for(int v=0;v<V;++v) {
      const float a=x[row*16384+base+v],b=weight[output*16384+base+v];
      sums[v]=Fma?__fmaf_rn(a,b,sums[v]):__fadd_rn(sums[v],__fmul_rn(a,b));
    }
  }
#pragma unroll
  for(int v=1;v<V;++v)sums[0]=__fadd_rn(sums[0],sums[v]);
  partial[lane]=sums[0];__syncthreads();
  for(int span=P/2;span>0;span>>=1) {
    if(lane<span)partial[lane]=__fadd_rn(partial[lane],partial[lane+span]);
    __syncthreads();
  }
  if(lane==0)mixes[row*24+output]=__fmul_rn(partial[0],inverse[row]);
}
__global__ void normalize(float *mixes,const float *inverse) {
  const int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<144*24)mixes[i]=__fmul_rn(mixes[i],inverse[i/24]);
}

// The observed exact vendor execution has two reduction levels: strided
// chunks with two independent FMA accumulators, then an ordered four-stripe
// publication. These probes isolate those laws without calling the vendor.
template<int Splits,int V,bool Fma> __global__ void split_projection(
    const float *x,const float *weight,float *partials) {
  const int row=blockIdx.x,output=blockIdx.y,split=blockIdx.z,lane=threadIdx.x;
  __shared__ float shared[128];float sums[V]={};
  for(int base=(split*128+lane)*V;base<16384;base+=Splits*128*V) {
#pragma unroll
    for(int v=0;v<V;++v)if(base+v<16384) {
      const float a=x[row*16384+base+v],b=weight[output*16384+base+v];
      sums[v]=Fma?__fmaf_rn(a,b,sums[v]):__fadd_rn(sums[v],__fmul_rn(a,b));
    }
  }
#pragma unroll
  for(int v=1;v<V;++v)sums[0]=__fadd_rn(sums[0],sums[v]);
  shared[lane]=sums[0];__syncthreads();
  for(int span=64;span>0;span>>=1) {
    if(lane<span)shared[lane]=__fadd_rn(shared[lane],shared[lane+span]);
    __syncthreads();
  }
  if(lane==0)partials[(row*24+output)*Splits+split]=shared[0];
}
template<int Splits,bool StripedFour> __global__ void publish_projection(
    const float *partials,const float *inverse,float *mixes) {
  const int row=blockIdx.x,output=blockIdx.y,lane=threadIdx.x;
  __shared__ float shared[32];
  shared[lane]=lane<Splits?partials[(row*24+output)*Splits+lane]:0.0f;
  __syncthreads();
  if(StripedFour) {
    if(lane<4) {
      float sum=0.0f;
      for(int i=lane;i<32;i+=4)sum=__fadd_rn(sum,shared[i]);
      shared[lane]=sum;
    }
    __syncthreads();
    if(lane==0) {
      float sum=0.0f;
      for(int i=0;i<4;++i)sum=__fadd_rn(sum,shared[i]);
      mixes[row*24+output]=__fmul_rn(sum,inverse[row]);
    }
  } else {
    for(int span=16;span>0;span>>=1) {
      if(lane<span)shared[lane]=__fadd_rn(shared[lane],shared[lane+span]);
      __syncthreads();
    }
    if(lane==0)mixes[row*24+output]=__fmul_rn(shared[0],inverse[row]);
  }
}
template<int Splits,int V,bool Fma,bool StripedFour> static void hierarchical(
    const float *x,const float *weights,const float *inverse,float *partials,
    float *mixes,const std::string &out,int rows) {
  const std::string label="hierarchical-s"+std::to_string(Splits)+"-v"+std::to_string(V)+
    (Fma?"-fma":"-separate")+(StripedFour?"-striped4":"-tree32");
  split_projection<Splits,V,Fma><<<dim3(rows,24,Splits),128>>>(x,weights,partials);
  publish_projection<Splits,StripedFour><<<dim3(rows,24),32>>>(partials,inverse,mixes);
  check(cudaGetLastError());check(cudaDeviceSynchronize());save(mixes,rows*24,out+"/"+label+".f32");
}

template<int P,int V,bool Fma> static void run(const float *x,const float *weights,
    const float *inverse,float *mixes,const std::string &out,int rows) {
  const std::string label="p"+std::to_string(P)+"-v"+std::to_string(V)+(Fma?"-fma":"-separate");
  projection<P,V,Fma><<<dim3(rows,24),P>>>(x,weights,inverse,mixes);
  check(cudaGetLastError());check(cudaDeviceSynchronize());save(mixes,rows*24,out+"/"+label+".f32");
}
template<int P> static void partition(const float *x,const float *w,const float *i,float *m,const std::string &out,int rows) {
#define ROUTE(V) run<P,V,false>(x,w,i,m,out,rows);run<P,V,true>(x,w,i,m,out,rows)
  ROUTE(1);ROUTE(2);ROUTE(4);ROUTE(8);
#undef ROUTE
}

int main(int argc,char **argv) {
  if(argc!=3&&argc!=6)return 2;const std::string input(argv[1]),out(argv[2]);
  const int rows=argc==6?std::atoi(argv[4]):144;
  const bool native_only=argc==6&&std::string(argv[5])=="--native-only";
  if(rows<1||rows>144||(argc==6&&(std::string(argv[3])!="--rows"||!native_only)))return 2;
  auto *x=load<__nv_bfloat16>(input+"/streams.bf16",144*16384);
  auto *weights=load<float>(input+"/function.f32",24*16384);
  auto *wide=allocate<float>(144*16384),*inverse=allocate<float>(144),*mixes=allocate<float>(144*24);
  auto *partials=allocate<float>(144*24*32);
  prepare<<<rows,512>>>(x,wide,inverse);check(cudaGetLastError());check(cudaDeviceSynchronize());
  save(inverse,rows,out+"/computed-rsqrt.f32");
#define PARTITION(P) partition<P>(wide,weights,inverse,mixes,out,rows)
  PARTITION(4);PARTITION(8);PARTITION(16);PARTITION(32);
  PARTITION(64);PARTITION(128);PARTITION(256);PARTITION(512);
#undef PARTITION
#define HIERARCHICAL(S,V,F,T) hierarchical<S,V,F,T>(wide,weights,inverse,partials,mixes,out,rows)
  HIERARCHICAL(18,2,false,false);HIERARCHICAL(18,2,false,true);
  HIERARCHICAL(18,2,true,false);HIERARCHICAL(18,2,true,true);
  HIERARCHICAL(16,2,true,true);HIERARCHICAL(24,2,true,true);
  HIERARCHICAL(18,1,true,true);HIERARCHICAL(18,4,true,true);
#undef HIERARCHICAL
  // Vendor library is an OFFLINE geometry control only. Same original F32
  // operands and one-row calls; never used by native serving or as a reference.
  if(!native_only) {
  cublasHandle_t handle;blas(cublasCreate(&handle));
  blas(cublasSetMathMode(handle,CUBLAS_PEDANTIC_MATH));const float alpha=1,beta=0;
  for(bool gemv:{true,false}) {
    for(int row=0;row<144;++row) {
      if(gemv)blas(cublasSgemv(handle,CUBLAS_OP_T,16384,24,&alpha,weights,16384,wide+row*16384,1,&beta,mixes+row*24,1));
      else blas(cublasSgemm(handle,CUBLAS_OP_T,CUBLAS_OP_N,24,1,16384,&alpha,weights,16384,wide+row*16384,16384,&beta,mixes+row*24,24));
    }
    normalize<<<14,256>>>(mixes,inverse);check(cudaGetLastError());check(cudaDeviceSynchronize());
    save(mixes,144*24,out+(gemv?"/cublas-gemv-pedantic.f32":"/cublas-gemm-pedantic.f32"));
  }
  blas(cublasDestroy(handle));
  }
  for(void *p:{(void*)x,(void*)weights,(void*)wide,(void*)inverse,(void*)mixes,(void*)partials})check(cudaFree(p));
  std::printf("control_dot_probe rows=%d outputs=24 width=16384 native_routes=72 vendor_routes=%d original_operands=true computed_rms=true offline_only=true release=balanced\n",rows,native_only?0:2);
}
