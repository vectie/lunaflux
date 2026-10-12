// Offline arithmetic isolation on captured original shared-expert operands.
// This is not a production kernel, schedule, or reference replacement.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <string>
#include <vector>
#include "shared_matched_kernels.cuh"

static void check(cudaError_t e) {
  if (e != cudaSuccess) { std::fprintf(stderr,"%s\n",cudaGetErrorString(e)); std::exit(2); }
}
template<class T> static T *allocate(size_t n) {
  T *p=nullptr; check(cudaMalloc(&p,n*sizeof(T))); check(cudaMemset(p,0,n*sizeof(T))); return p;
}
template<class T> static std::vector<T> read(const std::string &path,size_t n) {
  std::vector<T> values(n); auto *f=std::fopen(path.c_str(),"rb");
  if(!f){std::perror(path.c_str());std::exit(3);}
  const bool ok=std::fread(values.data(),sizeof(T),n,f)==n&&std::fgetc(f)==EOF;
  std::fclose(f); if(!ok)std::exit(3); return values;
}
template<class T> static T *upload(const std::vector<T> &values) {
  auto *p=allocate<T>(values.size()); check(cudaMemcpy(p,values.data(),values.size()*sizeof(T),cudaMemcpyHostToDevice)); return p;
}
template<class T> static void save(T *p,size_t n,const std::string &path) {
  std::vector<T> values(n); check(cudaMemcpy(values.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost));
  auto *f=std::fopen(path.c_str(),"wbx"); if(!f){std::perror(path.c_str());std::exit(4);}
  const bool ok=std::fwrite(values.data(),sizeof(T),n,f)==n; const bool closed=std::fclose(f)==0;
  if(!ok||!closed)std::exit(4);
}
static void report(const std::string &label,__nv_bfloat16 *p,const std::vector<__nv_bfloat16> &reference) {
  std::vector<__nv_bfloat16> values(reference.size());
  check(cudaMemcpy(values.data(),p,values.size()*sizeof(values[0]),cudaMemcpyDeviceToHost));
  size_t exact=0,nonfinite=0; double delta2=0,reference2=0;
  for(size_t i=0;i<values.size();++i) {
    const bool same=static_cast<__nv_bfloat16_raw>(values[i]).x==static_cast<__nv_bfloat16_raw>(reference[i]).x;
    if(same)++exact;
    const double a=__bfloat162float(values[i]),b=__bfloat162float(reference[i]);
    if(!std::isfinite(a))++nonfinite;
    delta2+=(a-b)*(a-b); reference2+=b*b;
    if(!same&&i==631)
      std::printf("frontier %s index=631 native=%.12g original=%.12g\n",label.c_str(),a,b);
  }
  std::printf("result %s exact=%zu/%zu nonfinite=%zu relative_l2_percent=%.12g\n",label.c_str(),exact,values.size(),nonfinite,100*std::sqrt(delta2/reference2));
}
static void observe_quantized(unsigned char *packed,const std::string &input,const std::string &name,int k) {
  const auto data=read<unsigned char>(input+"/"+name+"-quantized.fp8",k);
  const auto scales=read<unsigned char>(input+"/"+name+"-quantized.scale",k/128);
  std::vector<unsigned char> native(k+k/128);
  check(cudaMemcpy(native.data(),packed,native.size(),cudaMemcpyDeviceToHost));
  size_t same=0,same_scale=0;
  for(int i=0;i<k;++i)if(data[i]==native[i])++same;
  for(int i=0;i<k/128;++i)if(scales[i]==native[k+i])++same_scale;
  std::printf("quantized %s payload_exact=%zu/%d scale_exact=%zu/%d\n",name.c_str(),same,k,same_scale,k/128);
}

// Each output owns independent raw-operand partial sums. Scale products and
// outer publications are separate explicit rounding choices. Zero padding is
// physical only. The original logical group remains 128 columns.
template<int Partitions,bool InnerFma,bool OuterFma>
__global__ void block_dot(const unsigned char *x,const unsigned char *weight,const unsigned char *scales,
                          int k,int n,__nv_bfloat16 *output,float *raw_output,float *raw_blocks) {
  const int out=blockIdx.x*blockDim.x+threadIdx.x; if(out>=n)return;
  const int groups=k/128; float total=0;
  for(int group=0;group<groups;++group) {
    float partial[Partitions]={};
    for(int offset=0;offset<128;++offset) {
      const int column=group*128+offset,owner=offset%Partitions;
      const float a=lf_precision_e4(x[column]),b=lf_precision_e4(weight[out*k+column]);
      partial[owner]=InnerFma?__fmaf_rn(a,b,partial[owner]):__fadd_rn(partial[owner],__fmul_rn(a,b));
    }
    for(int stride=Partitions/2;stride>0;stride/=2)
      for(int owner=0;owner<stride;++owner)partial[owner]=__fadd_rn(partial[owner],partial[owner+stride]);
    const float scale=__fmul_rn(lf_precision_ue8(x[k+group]),lf_precision_ue8(scales[(out/128)*groups+group]));
    raw_blocks[out*groups+group]=partial[0];
    total=OuterFma?__fmaf_rn(partial[0],scale,total):__fadd_rn(total,__fmul_rn(partial[0],scale));
  }
  output[out]=__float2bfloat16_rn(total); raw_output[out]=total;
}
__global__ void exact_dot(const unsigned char *x,const unsigned char *weight,const unsigned char *scales,
                          int k,int n,__nv_bfloat16 *output,float *raw_output) {
  const int out=blockIdx.x*blockDim.x+threadIdx.x;if(out>=n)return;double total=0;
  for(int column=0;column<k;++column) {
    const double a=lf_precision_e4(x[column]),b=lf_precision_e4(weight[out*k+column]);
    const double scale=double(lf_precision_ue8(x[k+column/128]))*lf_precision_ue8(scales[(out/128)*(k/128)+column/128]);
    total+=a*b*scale;
  }
  output[out]=__float2bfloat16_rn(__double2float_rn(total));raw_output[out]=__double2float_rn(total);
}
__global__ void current_dot(const unsigned char *x,const unsigned char *weight,const unsigned char *scales,
                            int k,int n,__nv_bfloat16 *output,float *raw_output) {
  const int out=(blockIdx.x*blockDim.x+threadIdx.x)/32,lane=threadIdx.x%32;
  if(out>=n)return; float sum=0;
  for(int column=lane;column<k;column+=32) {
    const float a=__bfloat162float(__float2bfloat16_rn(__fmul_rn(lf_precision_e4(x[column]),lf_precision_ue8(x[k+column/128]))));
    const float b=__bfloat162float(__float2bfloat16_rn(__fmul_rn(lf_precision_e4(weight[out*k+column]),lf_precision_ue8(scales[(out/128)*(k/128)+column/128]))));
    sum=__fadd_rn(sum,__fmul_rn(a,b));
  }
  for(int stride=16;stride>0;stride/=2)sum=__fadd_rn(sum,__shfl_down_sync(0xffffffff,sum,stride));
  if(lane==0){output[out]=__float2bfloat16_rn(sum);raw_output[out]=sum;}
}
// PTX m16n8k32 E4M3 fragment ownership from NVIDIA's documented ABI.
// Only logical row zero is real; other physical rows are neutral padding.
template<bool OuterFma>
__global__ void matrix_dot(const unsigned char *x,const unsigned char *weight,const unsigned char *scales,
                          int k,int n,__nv_bfloat16 *output,float *raw_output,float *raw_blocks) {
  const int lane=threadIdx.x%32,g=lane/4,t=lane%4;
  const int tile=(blockIdx.x*(blockDim.x/32)+threadIdx.x/32)*8;
  if(tile>=n)return;float total[2]={};
  for(int group=0;group<k/128;++group) {
    float d[4]={};
    for(int sub=0;sub<128;sub+=32) {
      unsigned a[4]={},b[2]={};
      for(int byte=0;byte<4;++byte) {
        const int col=group*128+sub+t*4+byte;
        if(g==0){a[0]|=unsigned(x[col])<<(8*byte);a[2]|=unsigned(x[col+16])<<(8*byte);}
        b[0]|=unsigned(weight[(tile+g)*k+col])<<(8*byte);
        b[1]|=unsigned(weight[(tile+g)*k+col+16])<<(8*byte);
      }
      asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]) : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]));
    }
    if(g==0)for(int i=0;i<2;++i) {
      const int out=tile+t*2+i;
      const float scale=__fmul_rn(lf_precision_ue8(x[k+group]),lf_precision_ue8(scales[(out/128)*(k/128)+group]));
      total[i]=OuterFma?__fmaf_rn(d[i],scale,total[i]):__fadd_rn(total[i],__fmul_rn(d[i],scale));
      raw_blocks[out*(k/128)+group]=d[i];
    }
  }
  if(g==0)for(int i=0;i<2;++i){const int out=tile+t*2+i;output[out]=__float2bfloat16_rn(total[i]);raw_output[out]=total[i];}
}
template<int P> static void alternatives(const std::string &stage,const std::string &out,unsigned char *x,
                                       unsigned char *w,unsigned char *s,int k,int n,__nv_bfloat16 *y,float *f,
                                       float *blocks,const std::vector<float> &original_blocks,
                                       const std::vector<__nv_bfloat16> &reference) {
  for(int inner=0;inner<2;++inner)for(int outer=0;outer<2;++outer) {
    if(inner&&outer)block_dot<P,true,true><<<(n+127)/128,128>>>(x,w,s,k,n,y,f,blocks);
    else if(inner)block_dot<P,true,false><<<(n+127)/128,128>>>(x,w,s,k,n,y,f,blocks);
    else if(outer)block_dot<P,false,true><<<(n+127)/128,128>>>(x,w,s,k,n,y,f,blocks);
    else block_dot<P,false,false><<<(n+127)/128,128>>>(x,w,s,k,n,y,f,blocks);
    check(cudaGetLastError());check(cudaDeviceSynchronize());
    const std::string label=stage+"-p"+std::to_string(P)+"-inner"+std::to_string(inner)+"-outer"+std::to_string(outer);
    report(label,y,reference);save(y,n,out+"/"+label+".bf16");save(f,n,out+"/"+label+".f32");
    if(!original_blocks.empty()&&outer==0) {
      std::vector<float> local(n*(k/128));check(cudaMemcpy(local.data(),blocks,local.size()*4,cudaMemcpyDeviceToHost));
      size_t exact=0;for(size_t i=0;i<local.size();++i)if(local[i]==original_blocks[i])++exact;
      std::printf("blocks %s exact=%zu/%zu\n",label.c_str(),exact,local.size());
      for(int group=0;group<k/128;++group) {
        const int i=631*(k/128)+group;if(local[i]!=original_blocks[i])
          std::printf("block_frontier %s output=631 group=%d native=%.12g original=%.12g\n",label.c_str(),group,local[i],original_blocks[i]);
      }
    }
  }
}
int main(int argc,char **argv) {
  if(argc!=3&&argc!=4)return 2; const std::string input(argv[1]),out(argv[2]);
  const bool observed=argc==4&&std::string(argv[3])=="--observed";
  if(argc==4&&!observed)return 2;
  auto *counts=allocate<int>(5),*error=allocate<int>(1);const int live[5]={0,0,0,1,0};
  check(cudaMemcpy(counts,live,sizeof(live),cudaMemcpyHostToDevice));
  auto *x=upload(read<__nv_bfloat16>(input+"/input.bf16",4096));
  auto *activation=upload(read<__nv_bfloat16>(input+"/activation.bf16",2048));
  auto *q0=allocate<unsigned char>(shared_quantized_bytes[0]),*q1=allocate<unsigned char>(shared_quantized_bytes[1]);
  matched_shared_quantize_0<<<32,128>>>(counts,x,q0,error);
  matched_shared_quantize_1<<<16,128>>>(counts,activation,q1,error);
  check(cudaGetLastError());check(cudaDeviceSynchronize());
  save(q0,shared_quantized_bytes[0],out+"/input-packed.bin");save(q1,shared_quantized_bytes[1],out+"/activation-packed.bin");
  if(observed) {
    observe_quantized(q0,input,"input",4096);observe_quantized(q1,input,"activation",2048);
    const auto original=read<__nv_bfloat16>(input+"/output.bf16",4096);
    auto *matched=upload(read<__nv_bfloat16>(input+"/matched-output.bf16",4096));
    report("original-matched-input",matched,original);check(cudaFree(matched));
    auto *terminal=upload(read<__nv_bfloat16>(input+"/f32-terminal.bf16",4096));
    report("original-f32-terminal",terminal,original);check(cudaFree(terminal));
  }
  auto *y=allocate<__nv_bfloat16>(4096);auto *f=allocate<float>(4096),*blocks=allocate<float>(4096*32);
  for(int stage=0;stage<3;++stage) {
    const std::string name=stage==0?"gate":stage==1?"up":"output",weight=stage==0?"w1":stage==1?"w3":"w2";
    const int k=stage<2?4096:2048,n=stage<2?2048:4096;
    auto *w=upload(read<unsigned char>(input+"/"+weight+".fp8",8388608));
    auto *s=upload(read<unsigned char>(input+"/"+weight+".scale",512));
    auto *packed=stage<2?q0:q1;const auto reference=read<__nv_bfloat16>(input+"/"+name+".bf16",n);
    std::vector<float> original_blocks;
    if(observed&&stage==2) {
      original_blocks.resize(n*16);
      for(int group=0;group<16;++group) {
        const auto original=read<float>(input+"/down-unscaled-block-"+std::to_string(group)+".f32",n);
        for(int i=0;i<n;++i)original_blocks[i*16+group]=original[i];
      }
    }
    current_dot<<<(n+3)/4,128>>>(packed,w,s,k,n,y,f);check(cudaGetLastError());check(cudaDeviceSynchronize());
    report(name+"-current",y,reference);save(y,n,out+"/"+name+"-current.bf16");save(f,n,out+"/"+name+"-current.f32");
    if(observed&&stage==2) {
      const auto original_f32=read<float>(input+"/matched-output.f32",4096);
      std::vector<float> native(n);check(cudaMemcpy(native.data(),f,n*sizeof(float),cudaMemcpyDeviceToHost));
      std::printf("raw_frontier index=631 current=%.12g original=%.12g delta=%.12g\n",native[631],original_f32[631],native[631]-original_f32[631]);
    }
    exact_dot<<<(n+127)/128,128>>>(packed,w,s,k,n,y,f);check(cudaGetLastError());check(cudaDeviceSynchronize());
    report(name+"-double-diagnostic",y,reference);save(f,n,out+"/"+name+"-double-diagnostic.f32");
    for(int outer=0;outer<2;++outer) {
      if(outer)matrix_dot<true><<<(n+31)/32,128>>>(packed,w,s,k,n,y,f,blocks);
      else matrix_dot<false><<<(n+31)/32,128>>>(packed,w,s,k,n,y,f,blocks);
      check(cudaGetLastError());check(cudaDeviceSynchronize());
      const std::string label=name+"-matrix-outer"+std::to_string(outer);
      report(label,y,reference);save(y,n,out+"/"+label+".bf16");save(f,n,out+"/"+label+".f32");
      if(!original_blocks.empty()) {
        std::vector<float> local(n*(k/128));check(cudaMemcpy(local.data(),blocks,local.size()*4,cudaMemcpyDeviceToHost));
        size_t exact=0;for(size_t i=0;i<local.size();++i)if(local[i]==original_blocks[i])++exact;
        std::printf("blocks %s exact=%zu/%zu\n",label.c_str(),exact,local.size());
      }
    }
    alternatives<1>(name,out,packed,w,s,k,n,y,f,blocks,original_blocks,reference);
    alternatives<2>(name,out,packed,w,s,k,n,y,f,blocks,original_blocks,reference);
    alternatives<4>(name,out,packed,w,s,k,n,y,f,blocks,original_blocks,reference);
    alternatives<8>(name,out,packed,w,s,k,n,y,f,blocks,original_blocks,reference);
    alternatives<16>(name,out,packed,w,s,k,n,y,f,blocks,original_blocks,reference);
    alternatives<32>(name,out,packed,w,s,k,n,y,f,blocks,original_blocks,reference);
    alternatives<64>(name,out,packed,w,s,k,n,y,f,blocks,original_blocks,reference);
    alternatives<128>(name,out,packed,w,s,k,n,y,f,blocks,original_blocks,reference);
    check(cudaFree(w));check(cudaFree(s));
  }
  int failure=0;check(cudaMemcpy(&failure,error,4,cudaMemcpyDeviceToHost));
  for(void *p:{(void*)counts,(void*)error,(void*)x,(void*)activation,(void*)q0,(void*)q1,(void*)y,(void*)f,(void*)blocks})check(cudaFree(p));
  if(failure)return 5;std::puts("block_scaled_probe_complete release=balanced no_reference_substitution=true");
}
