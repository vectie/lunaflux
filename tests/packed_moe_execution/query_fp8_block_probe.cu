// Offline matched query projection: reuse the ACTUAL generated raw matrix
// block helper on original full-width inputs/weights. Norms are not changed.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <string>
#include <vector>
#include "query_matched_kernels.cuh"
#include "shared_matched_kernels.cuh"

static void check(cudaError_t e) {
  if(e!=cudaSuccess){std::fprintf(stderr,"%s\n",cudaGetErrorString(e));std::exit(2);}
}
template<class T>static T *allocate(size_t n) {
  T *p=nullptr;check(cudaMalloc(&p,n*sizeof(T)));check(cudaMemset(p,0,n*sizeof(T)));return p;
}
template<class T>static std::vector<T> read(const std::string &path,size_t n) {
  std::vector<T> values(n);auto *f=std::fopen(path.c_str(),"rb");if(!f){std::perror(path.c_str());std::exit(3);}
  const bool ok=std::fread(values.data(),sizeof(T),n,f)==n&&std::fgetc(f)==EOF;
  std::fclose(f);if(!ok)std::exit(3);return values;
}
template<class T>static T *upload(const std::vector<T> &values) {
  auto *p=allocate<T>(values.size());check(cudaMemcpy(p,values.data(),values.size()*sizeof(T),cudaMemcpyHostToDevice));return p;
}
template<int K>__global__ void quantize(const __nv_bfloat16 *input,unsigned char *packed,int rows) {
  const int group=blockIdx.x%(K/128),row=blockIdx.x/(K/128),col=group*128+threadIdx.x;
  if(row>=rows)return;__shared__ float scale;
  if(threadIdx.x==0){scale=lf_block_activation_scale(input,(unsigned long long)row*K+group*128);packed[rows*K+row*(K/128)+group]=__float_as_uint(scale)>>23;}
  __syncthreads();
  const float raw=__fdiv_rn(__bfloat162float(input[row*K+col]),scale);
  packed[row*K+col]=__nv_cvt_float_to_fp8(fminf(448.0f,fmaxf(-448.0f,raw)),__NV_SATFINITE,__NV_E4M3);
}
template<int K,int N>__global__ void project(const unsigned char *input,const unsigned char *weight,
                                           const unsigned char *scales,__nv_bfloat16 *output,int rows) {
  const int tile=(blockIdx.x*4+threadIdx.x/32)*8,row=blockIdx.y;
  if(tile>=N||row>=rows)return;float result[2];
  matched_shared_block_dot<K>(input+row*K,input+rows*K+row*(K/128),weight,scales,tile,result);
  const int lane=threadIdx.x%32;
  if(lane/4==0)for(int i=0;i<2;++i)output[row*N+tile+(lane%4)*2+i]=__float2bfloat16_rn(result[i]);
}
template<int K,int N>static void stage(const std::string &input,const std::string &out,const char *operand,
                                    const char *weight,const char *name,int rows) {
  auto *x=upload(read<__nv_bfloat16>(input+"/"+operand+".bf16",rows*K));
  auto *w=upload(read<unsigned char>(input+"/"+weight+".fp8",K*N));
  auto *s=upload(read<unsigned char>(input+"/"+weight+".scale",(K/128)*(N/128)));
  auto *q=allocate<unsigned char>(rows*(K+K/128));auto *y=allocate<__nv_bfloat16>(rows*N);
  quantize<K><<<rows*(K/128),128>>>(x,q,rows);
  project<K,N><<<dim3((N+31)/32,rows),128>>>(q,w,s,y,rows);
  check(cudaGetLastError());check(cudaDeviceSynchronize());
  const auto original=read<__nv_bfloat16>(input+"/"+name+".bf16",rows*N);
  std::vector<__nv_bfloat16> values(rows*N);check(cudaMemcpy(values.data(),y,values.size()*2,cudaMemcpyDeviceToHost));
  size_t exact=0,nonfinite=0;double delta=0,norm=0;
  for(size_t i=0;i<values.size();++i){
    if(static_cast<__nv_bfloat16_raw>(values[i]).x==static_cast<__nv_bfloat16_raw>(original[i]).x)++exact;
    const double a=__bfloat162float(values[i]),b=__bfloat162float(original[i]);
    if(!std::isfinite(a))++nonfinite;delta+=(a-b)*(a-b);norm+=b*b;
  }
  auto *file=std::fopen((out+"/"+name+".bf16").c_str(),"wbx");if(!file){std::perror("output");std::exit(4);}
  const bool written=std::fwrite(values.data(),2,values.size(),file)==values.size(),closed=std::fclose(file)==0;
  if(!written||!closed)std::exit(4);
  std::printf("matrix_query stage=%s exact=%zu/%zu nonfinite=%zu relative_l2_percent=%.12g\n",name,exact,values.size(),nonfinite,100*std::sqrt(delta/norm));
  for(void *p:{(void*)x,(void*)w,(void*)s,(void*)q,(void*)y})check(cudaFree(p));
}
int main(int argc,char **argv) {
  if(argc!=4)return 2;const int rows=std::atoi(argv[3]);if(rows<1||rows>144)return 2;
  const std::string input(argv[1]),out(argv[2]);
  stage<4096,1024>(input,out,"input","wq_a","q-a",rows);
  stage<1024,32768>(input,out,"q-norm","wq_b","q-b",rows);
  stage<4096,512>(input,out,"input","wkv","kv",rows);
  std::puts("matched_query_matrix_complete release=balanced norms_unchanged=true no_full_model_parity=true");
}
