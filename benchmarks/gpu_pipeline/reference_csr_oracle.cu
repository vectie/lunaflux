// Independent FP64 oracle for the production ten-pointer AOT wrapper.
// Exact-size CSR allocation (no padded tail), ragged phases, shuffled pages,
// inactive output preservation and repeatability. No interception shim.
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#define CK(x) do { auto e=(x); if(int(e)) { std::fprintf(stderr,"error=%d line=%d\n",int(e),__LINE__); std::exit(2); } } while(0)
template<class T> struct Buffer {
  T *p; size_t n;
  explicit Buffer(const std::vector<T>& h):n(h.size()) {
    CK(cudaMalloc(&p,n*sizeof(T))); CK(cudaMemcpy(p,h.data(),n*sizeof(T),cudaMemcpyHostToDevice));
  }
  ~Buffer(){CK(cudaFree(p));}
  std::vector<T> read(){std::vector<T> h(n);CK(cudaMemcpy(h.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost));return h;}
};
static float datum(size_t i,int salt) {return float(int((i*17+salt)%31)-15)/32.f;}
static void run(CUfunction f,unsigned shared,std::vector<int> queries,std::vector<int> lengths,int prefills,bool zero) {
  int b=queries.size(),tokens=0,pages=0;
  std::vector<int> offsets{0},po{0},positions,ids;
  for(int r=0;r<b;r++) {
    tokens+=queries[r];offsets.push_back(tokens);pages+=(lengths[r]+7)/8;po.push_back(pages);
    for(int q=0;q<queries[r];q++)positions.push_back(lengths[r]-queries[r]+q);
  }
  for(int p=0;p<pages;p++)ids.push_back(pages-1-p);
  Buffer<int> c({prefills,b-prefills,b,tokens,pages}),pos(positions),off(offsets),len(lengths),pageoff(po),pageids(ids);
  std::vector<__nv_bfloat16> x(size_t(tokens)*4096),k(size_t(pages)*8192),v(k.size());
  for(size_t i=0;i<x.size();i++)x[i]=__float2bfloat16(zero?0:datum(i,3));
  for(size_t i=0;i<k.size();i++){k[i]=__float2bfloat16(zero?0:datum(i,29));v[i]=__float2bfloat16(datum(i,31));}
  Buffer<__nv_bfloat16> dx(x),dk(k),dv(v);
  std::vector<__nv_bfloat16> poison(size_t(tokens+7)*2048,__float2bfloat16(1024));
  Buffer<__nv_bfloat16> out(poison);
  void* args[]={&c.p,&pos.p,&off.p,&len.p,&pageoff.p,&pageids.p,&dx.p,&out.p,&dk.p,&dv.p};
  CK(cuLaunchKernel(f,64,16,1,128,1,1,shared,nullptr,args,nullptr));CK(cudaDeviceSynchronize());
  auto actual=out.read();size_t checked=0;double maxerr=0;
  const int written_rows=prefills>0?b:0;
  for(int r=0;r<written_rows;r++)for(int t=offsets[r];t<offsets[r+1];t++)for(int h=0;h<16;h++) {
    int history=positions[t]+1;std::vector<double> scores(history);double maximum=-INFINITY,denom=0;
    auto at=[&](int p,int d){return size_t(ids[po[r]+p/8])*8192+(p%8)*1024+(h/2)*128+d;};
    for(int p=0;p<history;p++) {
      double dot=0;if(!zero)for(int d=0;d<128;d++)dot+=double(__bfloat162float(x[size_t(t)*4096+h*128+d]))*__bfloat162float(k[at(p,d)]);
      scores[p]=dot/std::sqrt(128.0);maximum=std::max(maximum,scores[p]);
    }
    for(auto& s:scores){s=std::exp(s-maximum);denom+=s;}
    for(int d=0;d<128;d++) {
      double expected=0;for(int p=0;p<history;p++)expected+=scores[p]*__bfloat162float(v[at(p,d)]);expected/=denom;
      double got=__bfloat162float(actual[size_t(t)*2048+h*128+d]),err=std::abs(got-expected);maxerr=std::max(maxerr,err);
      if(!std::isfinite(got)||err>.01+.02*std::abs(expected)){std::fprintf(stderr,"mismatch row=%d token=%d head=%d dim=%d got=%g expected=%g\n",r,t,h,d,got,expected);std::exit(3);}++checked;
    }
  }
  // The mixed implementation is the sole writer; pure decode stays untouched.
  for(size_t i=size_t(offsets[written_rows])*2048;i<actual.size();i++)
    if(std::memcmp(&actual[i],&poison[i],2))std::exit(4);
  auto afterx=dx.read(),afterk=dk.read(),afterv=dv.read();
  if(std::memcmp(afterx.data(),x.data(),x.size()*2)||std::memcmp(afterk.data(),k.data(),k.size()*2)||std::memcmp(afterv.data(),v.data(),v.size()*2))std::exit(5);
  CK(cuLaunchKernel(f,64,16,1,128,1,1,shared,nullptr,args,nullptr));CK(cudaDeviceSynchronize());auto repeated=out.read();
  if(std::memcmp(actual.data(),repeated.data(),actual.size()*2))std::exit(6);
  std::printf("prefills=%d rows=%d tokens=%d csr_entries=%d checked=%zu max_abs=%g inactive_unchanged=1 readonly=1 repeatable=1\n",prefills,b,tokens,pages,checked,maxerr);
}
int main(int argc,char** argv) {
  if(argc!=2 && argc!=3)return 1;
  const unsigned shared=argc==3?std::atoi(argv[2]):81920;
  if(shared!=81920 && shared!=49152)return 1;
  CK(cudaSetDevice(0));CK(cudaFree(nullptr));
  CUmodule module;CUfunction function;CK(cuModuleLoad(&module,argv[1]));
  CK(cuModuleGetFunction(&function,module,"lunaflux_attention_mixed_flashattention_bf16_exp2_v1"));
  CK(cuFuncSetAttribute(function,CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,shared));
  for(int history:{1,7,8,9,15,16,17,63,64,65,127,128,129})run(function,shared,{1},{history},1,false);
  run(function,shared,{63,7,1},{129,15,9},2,false);
  run(function,shared,{1,1,1},{1,9,129},0,false);
  run(function,shared,{64,65},{127,129},2,false);
  // Long-history oracle remains bounded; page 1024 and the CSR end are exact.
  run(function,shared,{8,1},{8193,8255},1,true);
  CK(cuModuleUnload(module));CK(cudaDeviceReset());std::puts("outcome=passed");
}
