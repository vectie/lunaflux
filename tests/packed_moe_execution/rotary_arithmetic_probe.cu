// Offline counterfactuals on identical inputs: production arithmetic is unchanged.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <string>
#include <vector>
#include "rotary_arithmetic_kernels.cuh"

static void check(cudaError_t code) {
  if(code!=cudaSuccess){std::fprintf(stderr,"%s\n",cudaGetErrorString(code));std::exit(2);}
}
template<class T> static std::vector<T> read(const std::string &path,size_t n) {
  auto *file=std::fopen(path.c_str(),"rb");if(!file)std::exit(3);
  std::vector<T> values(n);
  const bool exact=std::fread(values.data(),sizeof(T),n,file)==n&&std::fgetc(file)==EOF;
  const bool closed=std::fclose(file)==0;if(!exact||!closed)std::exit(3);return values;
}
template<class T> static T *upload(const std::vector<T> &values) {
  T *pointer=nullptr;check(cudaMalloc(&pointer,values.size()*sizeof(T)));
  check(cudaMemcpy(pointer,values.data(),values.size()*sizeof(T),cudaMemcpyHostToDevice));return pointer;
}

template<bool Literal,int Rounding> __global__ void intervention(
    const uint16_t *input,const float2 *frequencies,uint16_t *output) {
  const int row=blockIdx.x,head=blockIdx.y;
  const size_t base=(size_t(row)*64+head)*512;
  for(int col=threadIdx.x;col<448;col+=256)output[base+col]=input[base+col];
  for(int pair=threadIdx.x;pair<32;pair+=256) {
    float c,s;
    if constexpr(Literal) {
      const float2 z=frequencies[row*32+pair];c=z.x;s=-z.y;
    } else {
      const float angle=__fmul_rn(float(row),generated_frequency(pair));c=cosf(angle);s=-sinf(angle);
    }
    const size_t i=base+448+pair*2;
    const auto *source=reinterpret_cast<const __nv_bfloat16*>(input);
    auto *target=reinterpret_cast<__nv_bfloat16*>(output);
    const float l=__bfloat162float(source[i]),r=__bfloat162float(source[i+1]);
    float real,imag;
    if constexpr(Rounding==0) {
      real=__fadd_rn(__fmul_rn(l,c),-__fmul_rn(r,s));
      imag=__fadd_rn(__fmul_rn(l,s),__fmul_rn(r,c));
    } else {
      if constexpr(Rounding==1||Rounding==3)real=__fmaf_rn(l,c,-__fmul_rn(r,s));
      else real=__fmaf_rn(-r,s,__fmul_rn(l,c));
      if constexpr(Rounding==1||Rounding==4)imag=__fmaf_rn(l,s,__fmul_rn(r,c));
      else imag=__fmaf_rn(r,c,__fmul_rn(l,s));
    }
    target[i]=__float2bfloat16_rn(real);target[i+1]=__float2bfloat16_rn(imag);
  }
}

static void report(const char *name,const std::vector<uint16_t> &values,
                   const std::vector<uint16_t> &reference,const std::string &root) {
  size_t exact=0,prefix_differences=0,nonfinite=0;
  double error=0,norm=0,suffix_error=0,suffix_norm=0,maximum=0;
  for(size_t i=0;i<values.size();++i) {
    const bool same=values[i]==reference[i];exact+=same;
    prefix_differences+=!same&&i%512<448;
    const double x=__bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&values[i]));
    const double y=__bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&reference[i]));
    if(!std::isfinite(x)||!std::isfinite(y)){++nonfinite;continue;}
    const double d=x-y;error+=d*d;norm+=y*y;maximum=std::fmax(maximum,std::fabs(d));
    if(i%512>=448){suffix_error+=d*d;suffix_norm+=y*y;}
  }
  auto *file=std::fopen((root+"/"+name+".bf16").c_str(),"wbx");if(!file)std::exit(4);
  const bool written=std::fwrite(values.data(),2,values.size(),file)==values.size();
  const bool closed=std::fclose(file)==0;if(!written||!closed)std::exit(4);
  std::printf("%s\t%zu\t%zu\t%zu\t%zu\t%.17g\t%.17g\t%.17g\n",name,values.size(),exact,
    prefix_differences,nonfinite,std::sqrt(error/norm),std::sqrt(suffix_error/suffix_norm),maximum);
}

template<bool Literal,int Rounding> static void run(const char *name,uint16_t *input,
    float2 *frequencies,uint16_t *output,const std::vector<uint16_t> &reference,
    const std::string &root) {
  intervention<Literal,Rounding><<<dim3(144,64),256>>>(input,frequencies,output);
  check(cudaGetLastError());check(cudaDeviceSynchronize());
  std::vector<uint16_t> values(reference.size());
  check(cudaMemcpy(values.data(),output,values.size()*2,cudaMemcpyDeviceToHost));report(name,values,reference,root);
}

int main(int argc,char **argv) {
  if(argc!=3)return 2;
  const std::string input_root(argv[1]),output_root(argv[2]);
  const size_t n=144*64*512;
  const auto source=read<uint16_t>(input_root+"/attention.bf16",n);
  const auto reference=read<uint16_t>(input_root+"/inverse.bf16",n);
  const auto baseline=read<uint16_t>(input_root+"/native-inverse.bf16",n);
  const auto frequency=read<float2>(input_root+"/frequencies.complex64",144*32);
  auto *device_input=upload(source),*device_output=upload(std::vector<uint16_t>(n,0));
  auto *device_frequencies=upload(frequency);
  auto *counts=upload(std::vector<int>{0,0,1,144,0});
  std::vector<int> positions(144);for(int i=0;i<144;++i)positions[i]=i;
  auto *device_positions=upload(positions);
  production_inverse<<<dim3(144,64),256>>>(counts,device_positions,device_input,device_output);
  check(cudaGetLastError());check(cudaDeviceSynchronize());
  std::vector<uint16_t> produced(n);
  check(cudaMemcpy(produced.data(),device_output,n*2,cudaMemcpyDeviceToHost));
  const bool baseline_equal=produced==baseline;
  std::puts("route\telements\texact_bf16\tprefix_differences\tnonfinite\trelative_l2\tsuffix_relative_l2\tmaximum_absolute_error");
  report("production",produced,reference,output_root);
  contracted_inverse<<<dim3(144,64),256>>>(counts,device_positions,device_input,device_output);
  check(cudaGetLastError());check(cudaDeviceSynchronize());
  std::vector<uint16_t> contracted(n);
  check(cudaMemcpy(contracted.data(),device_output,n*2,cudaMemcpyDeviceToHost));
  report("lowered_complex",contracted,reference,output_root);
  run<false,0>("computed_separate",device_input,device_frequencies,device_output,reference,output_root);
  run<true,0>("literal_separate",device_input,device_frequencies,device_output,reference,output_root);
  run<false,1>("computed_fma_left",device_input,device_frequencies,device_output,reference,output_root);
  run<false,2>("computed_fma_right",device_input,device_frequencies,device_output,reference,output_root);
  run<false,3>("computed_fma_left_right",device_input,device_frequencies,device_output,reference,output_root);
  run<false,4>("computed_fma_right_left",device_input,device_frequencies,device_output,reference,output_root);
  run<true,1>("literal_fma_left",device_input,device_frequencies,device_output,reference,output_root);
  run<true,2>("literal_fma_right",device_input,device_frequencies,device_output,reference,output_root);
  run<true,3>("literal_fma_left_right",device_input,device_frequencies,device_output,reference,output_root);
  run<true,4>("literal_fma_right_left",device_input,device_frequencies,device_output,reference,output_root);
  for(void *pointer:{(void*)device_input,(void*)device_output,(void*)device_frequencies,(void*)counts,(void*)device_positions})check(cudaFree(pointer));
  if(!baseline_equal){std::fprintf(stderr,"actual renderer does not reproduce retained production output\n");return 5;}
  std::puts("retained_production_bitwise=true release=balanced serving_changes=false");
}
