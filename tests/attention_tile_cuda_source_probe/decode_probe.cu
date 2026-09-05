// Real-device test of compiler map/fold/merge, including empty partitions and
// mixed rows. No model-specific weights or production runtime dependencies.
#include <cuda.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static void ck(CUresult r) {
  if (r != CUDA_SUCCESS) { const char *s = nullptr; cuGetErrorString(r, &s); std::fprintf(stderr, "%s\n", s); std::exit(2); }
}
static float fp(uint16_t x) { uint32_t u = uint32_t(x) << 16; float f; std::memcpy(&f, &u, 4); return f; }
static uint16_t bf(float f) { uint32_t u; std::memcpy(&u, &f, 4); u += 0x7fff + ((u >> 16) & 1); return uint16_t(u >> 16); }
template<class T> static CUdeviceptr upload(const std::vector<T>& a) {
  CUdeviceptr p; ck(cuMemAlloc(&p, a.size()*sizeof(T))); ck(cuMemcpyHtoD(p, a.data(), a.size()*sizeof(T))); return p;
}
struct Kernel {
  CUmodule module; CUfunction partial, merge; bool split; int heads, threads, shared;
  Kernel(const char* path, bool s, bool grouped): split(s), heads(grouped?8:16), threads(grouped?256:32), shared(grouped?34076:0) {
    ck(cuModuleLoad(&module, path)); ck(cuModuleGetFunction(&partial, module, s?"decode_partial":"decode_baseline"));
    if(s) ck(cuModuleGetFunction(&merge,module,"decode_merge"));
  }
  void launch(CUdeviceptr* d) {
    // counts, positions, row offsets, lengths, page offsets, page ids, qkv,
    // output, key cache, value cache, workspace.
    void* args[]={d,d+1,d+2,d+3,d+4,d+5,d+6,split?d+8:d+7,split?d+9:d+8,split?d+10:d+9};
    ck(cuLaunchKernel(partial,32,heads,split?8:1,threads,1,1,shared,0,args,nullptr));
    if(split) { void* merge_args[]={d,d+2,d+10,d+7}; ck(cuLaunchKernel(merge,32,16,1,threads,1,1,0,0,merge_args,nullptr)); }
  }
};
static void run_case(Kernel* kernels,int context,int batch,bool mixed,int repeats) {
  const int prefill=mixed?1:0, rows=batch+prefill, prefix=mixed?3:0, tokens=batch+prefix;
  const int pages=(context+15)/16;
  std::vector<int> counts={prefill,batch,rows,tokens,batch*pages}, positions(tokens,0), offsets(rows+1), lengths(rows), page_offsets(rows+1), ids;
  if(mixed) { offsets[1]=prefix; lengths[0]=prefix; page_offsets[1]=1; ids.push_back(0); }
  for(int b=0;b<batch;++b) {
    int row=prefill+b, token=prefix+b; offsets[row]=token; offsets[row+1]=token+1; positions[token]=context-1; lengths[row]=context;
    page_offsets[row]=int(ids.size());
    for(int p=0;p<pages;++p) ids.push_back((b*pages+p+prefill)*3%8192);
    page_offsets[row+1]=int(ids.size());
  }
  counts[4]=int(ids.size());
  std::vector<uint16_t> q(tokens*4096), k(8192ULL*16384), v(k.size()), output(tokens*2048,bf(-99.0f));
  for(size_t i=0;i<q.size();++i) q[i]=bf(float(int((i*17)%127)-63)/64);
  for(size_t i=0;i<k.size();++i) { k[i]=bf(float(int((i*13)%113)-56)/64); v[i]=bf(float(int((i*19)%109)-54)/64); }
  std::vector<float> workspace(32*16*8*130, NAN);
  CUdeviceptr d[]={upload(counts),upload(positions),upload(offsets),upload(lengths),upload(page_offsets),upload(ids),upload(q),upload(output),upload(k),upload(v),upload(workspace)};
  // Double precision stable CPU oracle reads exactly the paged K/V layout.
  std::vector<float> expected(batch*2048);
  for(int b=0;b<batch;++b) for(int h=0;h<16;++h) {
    std::vector<double> scores(context); double maximum=-INFINITY, denominator=0;
    for(int p=0;p<context;++p) {
      size_t base=size_t(ids[page_offsets[prefill+b]+p/16])*16384+(p%16)*1024+(h/2)*128;
      double dot=0; for(int c=0;c<128;++c) dot+=double(fp(q[(prefix+b)*4096+h*128+c]))*fp(k[base+c]);
      scores[p]=dot/std::sqrt(128.0); maximum=std::max(maximum,scores[p]);
    }
    for(double& score:scores) { score=std::exp(score-maximum); denominator+=score; }
    for(int c=0;c<128;++c) {
      double sum=0; for(int p=0;p<context;++p) { size_t base=size_t(ids[page_offsets[prefill+b]+p/16])*16384+(p%16)*1024+(h/2)*128; sum+=scores[p]*fp(v[base+c]); }
      expected[b*2048+h*128+c]=float(sum/denominator);
    }
  }
  for(int which=0;which<3;++which) {
    ck(cuMemcpyHtoD(d[7],output.data(),output.size()*2)); kernels[which].launch(d); ck(cuCtxSynchronize());
    std::vector<uint16_t> got(output.size()); ck(cuMemcpyDtoH(got.data(),d[7],got.size()*2));
    float error=0;
    for(int i=0;i<batch*2048;++i) { float actual=fp(got[prefix*2048+i]); if(!std::isfinite(actual)) { std::fprintf(stderr,"nonfinite case=%d batch=%d kernel=%d i=%d\n",context,batch,which,i); std::exit(3); } error=std::max(error,std::fabs(actual-expected[i])); }
    if(error>0.004f) { std::fprintf(stderr,"oracle mismatch %f\n",error); std::exit(3); }
    for(int i=0;i<prefix*2048;++i) if(got[i]!=output[i]) { std::fprintf(stderr,"prefill row overwritten\n"); std::exit(3); }
    for(int n=0;n<10;++n) kernels[which].launch(d);
    CUevent start,end; ck(cuEventCreate(&start,0)); ck(cuEventCreate(&end,0)); ck(cuEventRecord(start,0));
    for(int n=0;n<repeats;++n) kernels[which].launch(d);
    ck(cuEventRecord(end,0)); ck(cuEventSynchronize(end)); float ms; ck(cuEventElapsedTime(&ms,start,end));
    std::printf("context=%d batch=%d mixed=%d kernel=%s us=%.6f max_abs_error=%.8f\n",context,batch,int(mixed),which==0?"baseline":which==1?"direct-split8":"grouped-split8",ms*1000/repeats,error);
    ck(cuEventDestroy(start)); ck(cuEventDestroy(end));
  }
  // Both compiler partition paths must leave persistent cache bytes untouched.
  std::vector<uint16_t> verify(k.size()); ck(cuMemcpyDtoH(verify.data(),d[8],verify.size()*2)); if(verify!=k) std::exit(4);
  ck(cuMemcpyDtoH(verify.data(),d[9],verify.size()*2)); if(verify!=v) std::exit(4);
  for(CUdeviceptr p:d) ck(cuMemFree(p));
}
int main(int argc,char**argv) {
  if(argc!=5) return 2;
  ck(cuInit(0)); CUdevice dev; ck(cuDeviceGet(&dev,0)); CUcontext ctx; ck(cuCtxCreate(&ctx,nullptr,0,dev));
  Kernel kernels[]={{argv[1],false,true},{argv[2],true,false},{argv[3],true,true}};
  int repeats=std::atoi(argv[4]); if(repeats<=0) return 2;
  for(int context:{1,59,128,256,512,1528,4096}) run_case(kernels,context,1,false,repeats);
  run_case(kernels,256,2,true,repeats); run_case(kernels,1528,8,false,repeats);
  for(auto& kernel:kernels) ck(cuModuleUnload(kernel.module)); ck(cuCtxDestroy(ctx));
  std::puts("correctness=passed kv_unchanged=true resources_released=true");
}
