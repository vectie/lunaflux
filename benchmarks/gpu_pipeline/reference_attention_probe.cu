// Offline native qualification only. No host ML framework or runtime JIT.
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CK(x) do { auto e=(x); if(e!=0) { \
  std::fprintf(stderr,"CUDA error=%d line=%d\n",int(e),__LINE__); std::exit(2); \
} } while(0)

#include "attention_referee.h"

struct Buffer {
  void *p=nullptr;
  size_t n;
  explicit Buffer(size_t bytes):n(bytes) {
    CK(cudaMalloc(&p,n)); CK(cudaMemset(p,0,n));
  }
  ~Buffer() { CK(cudaFree(p)); }
  Buffer(const Buffer&)=delete;
  Buffer& operator=(const Buffer&)=delete;
  void ints(const std::vector<int>& values) {
    if(values.size()*sizeof(int)>n) std::exit(3);
    CK(cudaMemcpy(p,values.data(),values.size()*sizeof(int),cudaMemcpyHostToDevice));
  }
  void fill(int salt) {
    std::vector<__nv_bfloat16> values(n/2);
    for(size_t i=0;i<values.size();++i)
      values[i]=__float2bfloat16(float(int((i*17+salt)%31)-15)/32.f);
    CK(cudaMemcpy(p,values.data(),n,cudaMemcpyHostToDevice));
  }
  std::vector<unsigned char> read() const {
    std::vector<unsigned char> values(n);
    CK(cudaMemcpy(values.data(),p,n,cudaMemcpyDeviceToHost));
    return values;
  }
};

struct Kernel {
  CUmodule module{};
  CUfunction function{};
  unsigned gx,gy,bx,by,bz,shared;
  Kernel(const char *path,const char *symbol,unsigned x,unsigned y,
         unsigned tx,unsigned ty,unsigned tz,unsigned bytes)
    :gx(x),gy(y),bx(tx),by(ty),bz(tz),shared(bytes) {
    CK(cuModuleLoad(&module,path));
    CK(cuModuleGetFunction(&function,module,symbol));
    CK(cuFuncSetAttribute(function,CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,shared));
    int regs=0;
    CK(cuFuncGetAttribute(&regs,CU_FUNC_ATTRIBUTE_NUM_REGS,function));
    std::printf("symbol=%s registers=%d shared=%u block=%u,%u,%u grid=%u,%u,1\n",
                symbol,regs,shared,bx,by,bz,gx,gy);
  }
  ~Kernel() { CK(cuModuleUnload(module)); }
  Kernel(const Kernel&)=delete;
  Kernel& operator=(const Kernel&)=delete;
  void launch(void **args) {
    CK(cuLaunchKernel(function,gx,gy,1,bx,by,bz,shared,nullptr,args,nullptr));
  }
  float time(void **args,int repeats) {
    cudaEvent_t begin,end;
    CK(cudaEventCreate(&begin)); CK(cudaEventCreate(&end));
    for(int i=0;i<3;++i) launch(args);
    CK(cudaDeviceSynchronize());
    CK(cudaEventRecord(begin));
    for(int i=0;i<repeats;++i) launch(args);
    CK(cudaEventRecord(end)); CK(cudaEventSynchronize(end));
    float ms=0;
    CK(cudaEventElapsedTime(&ms,begin,end));
    CK(cudaEventDestroy(begin)); CK(cudaEventDestroy(end));
    return ms*1000/repeats;
  }
};

int main(int argc,char **argv) {
  // BACKEND CUBIN ROWS QUERY_TOKENS_PER_ROW HISTORY FRAGMENTED REPEATS
  // Optional: --compiler SELECTED_CUBIN EXACT_SELECTED_SYMBOL (owned8 recipe).
  // Paired terminal schedules: --reference OLD_CUBIN OLD_SHARED NEW_SHARED.
  if(argc!=8 && argc!=11 && argc!=12) return 1;
  const bool paired_compiler=argc==11,paired_reference=argc==12;
  const bool paired=paired_compiler||paired_reference;
  if(paired_compiler && std::string(argv[8])!="--compiler") return 1;
  if(paired_reference && std::string(argv[8])!="--reference") return 1;
  std::string backend=argv[1];
  const bool mixed=backend=="flash-mixed";
  const bool prefill=backend=="flash-prefill" || mixed;
  const bool infer=backend=="infer-decode";
  if(!prefill && !infer && backend!="flash-decode") return 1;
  int rows=std::atoi(argv[3]),q=std::atoi(argv[4]),history=std::atoi(argv[5]);
  int fragmented=std::atoi(argv[6]),repeats=std::atoi(argv[7]);
  if(rows<1 || rows>32 || q<1 || q>2048 || rows*q>2048 ||
     history<q || history>65536 || (!prefill && q!=1) ||
     (fragmented!=0 && fragmented!=1) || repeats<1 || repeats>100 ||
     rows*((history+7)/8)>9216) return 1;
  const int tokens=rows*q;
  CK(cudaSetDevice(0)); CK(cudaFree(nullptr));
  {
    Buffer counts(20),positions(2048*4),offsets(33*4),lengths(32*4);
    Buffer page_offsets(33*4),page_indices(9216*4);
    Buffer input(2048*4096*2),output(2048*2048*2);
    Buffer keys(size_t(9216)*8192*2),values(size_t(9216)*8192*2);
    input.fill(3); keys.fill(29); values.fill(31);
    std::vector<int> offsets_host{0},lengths_host,page_offsets_host{0},pages,positions_host;
    for(int row=0;row<rows;++row) {
      const int context=history;
      lengths_host.push_back(context);
      for(int i=0;i<q;++i) positions_host.push_back(context-q+i);
      offsets_host.push_back((row+1)*q);
      for(int i=0;i<(context+7)/8;++i) {
        int index=int(pages.size());
        if(index>=9216) return 1;
        pages.push_back(fragmented ? (index*37)%9216 : index);
      }
      page_offsets_host.push_back(int(pages.size()));
    }
    counts.ints({prefill?rows:0,prefill?0:rows,rows,tokens,int(pages.size())});
    positions.ints(positions_host); offsets.ints(offsets_host);
    lengths.ints(lengths_host); page_offsets.ints(page_offsets_host); page_indices.ints(pages);
    const std::string symbol="lunaflux_attention_"+
      std::string(mixed?"mixed":prefill?"prefill":"decode")+
      (infer?"_flashinfer_f32_exp2_v1":"_flashattention_bf16_exp2_v1");
    const unsigned gx=prefill ? 32+(2048-32+63)/64 : 32;
    const unsigned shared=paired_reference?std::atoi(argv[11]):infer?9216:81920;
    const unsigned old_shared=paired_reference?std::atoi(argv[10]):33040;
    if(paired_reference && (infer || shared<16384 || shared>98304 ||
                           old_shared<16384 || old_shared>98304)) return 1;
    Kernel kernel(argv[2],symbol.c_str(),gx,prefill?16:8,
                  infer?16:128,infer?2:1,infer?4:1,shared);
    void *args[]={&counts.p,&positions.p,&offsets.p,&lengths.p,&page_offsets.p,
                  &page_indices.p,&input.p,&output.p,&keys.p,&values.p};
    const auto original_input=input.read(),original_keys=keys.read(),original_values=values.read();
    CK(cudaMemset(output.p,0xa5,output.n));
    kernel.launch(args); CK(cudaDeviceSynchronize());
    auto result=output.read();
    for(size_t i=0;i<size_t(tokens)*2048;++i)
      if(!std::isfinite(read_bf16(result,i))) {
        std::fprintf(stderr,"non-finite output at=%zu\n",i); return 4;
      }
    for(size_t i=size_t(tokens)*2048*2;i<result.size();++i)
      if(result[i]!=0xa5) { std::fprintf(stderr,"inactive output modified\n"); return 4; }
    if(original_input!=input.read() || original_keys!=keys.read() || original_values!=values.read()) {
      std::fprintf(stderr,"read-only operand modified\n"); return 4;
    }
    check_attention_referee(original_input,original_keys,original_values,result,
                            positions_host,offsets_host,page_offsets_host,pages);
    kernel.launch(args); CK(cudaDeviceSynchronize());
    if(result!=output.read()) { std::fprintf(stderr,"non-deterministic output\n"); return 4; }
    if(paired) {
      if(paired_compiler && prefill) return 1;
      Kernel compiler(argv[9],paired_reference?symbol.c_str():argv[10],
                      paired_reference?gx:32,paired_reference&&prefill?16:8,
                      paired_reference?128:64,1,1,old_shared);
      compiler.launch(args); CK(cudaDeviceSynchronize());
      auto compiled_result=output.read();
      check_attention_referee(original_input,original_keys,original_values,compiled_result,
                              positions_host,offsets_host,page_offsets_host,pages);
      float maxabs=0;
      for(size_t i=0;i<size_t(tokens)*2048;++i) {
        float a=read_bf16(result,i),b=read_bf16(compiled_result,i);
        if(!std::isfinite(b)) return 4;
        maxabs=std::fmax(maxabs,std::fabs(a-b));
      }
      if(maxabs>0.003f) { std::fprintf(stderr,"paired numerical error=%g\n",maxabs); return 4; }
      for(int trial=0;trial<5;++trial) {
        float old_us,new_us;
        if(trial%2) { new_us=kernel.time(args,repeats); old_us=compiler.time(args,repeats); }
        else { old_us=compiler.time(args,repeats); new_us=kernel.time(args,repeats); }
        std::printf("backend=%s rows=%d queries=%d history=%d fragmented=%d trial=%d compiler_us=%.6f reference_us=%.6f maxabs=%g\n",
                    backend.c_str(),rows,q,history,fragmented,trial,old_us,new_us,maxabs);
      }
    } else {
      for(int trial=0;trial<5;++trial)
        std::printf("backend=%s rows=%d queries=%d history=%d fragmented=%d trial=%d us=%.6f\n",
                    backend.c_str(),rows,q,history,fragmented,trial,kernel.time(args,repeats));
    }
    std::printf("outcome=passed scope=BF16-native-reference-qualification-not-serving\n");
  }
  CK(cudaDeviceReset());
}
