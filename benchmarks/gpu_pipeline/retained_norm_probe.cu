// Offline exact-order A/B check. No model or framework runtime dependency.
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cmath>
#define CK(x) do { int e=int(x); if(e) { fprintf(stderr,"CUDA %d line %d\n",e,__LINE__); exit(2); } } while(0)
struct Buffer {
  void *p=nullptr; size_t n;
  explicit Buffer(size_t size):n(size) { CK(cudaMalloc(&p,n)); }
  ~Buffer() { CK(cudaFree(p)); }
  void put(const std::vector<__nv_bfloat16>& v) { CK(cudaMemcpy(p,v.data(),n,cudaMemcpyHostToDevice)); }
  std::vector<__nv_bfloat16> get() { std::vector<__nv_bfloat16> v(n/2); CK(cudaMemcpy(v.data(),p,n,cudaMemcpyDeviceToHost)); return v; }
};
struct Module {
  CUmodule m; CUfunction f;
  explicit Module(const char *path, bool production) {
    CK(cuModuleLoad(&m,path));
    CK(cuModuleGetFunction(&f,m,production ? "lunaflux_fused_residual_rmsnorm_bf16_block128_tree_production_v2" : "lf_norm"));
  }
  ~Module() { CK(cuModuleUnload(m)); }
  void run(unsigned rows, void **args) { CK(cuLaunchKernel(f,rows,1,1,128,1,1,512,nullptr,args,nullptr)); }
  float time(unsigned rows,void **args) {
    cudaGraph_t graph; cudaGraphExec_t exec; cudaStream_t stream;
    CK(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    CK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
    for(int i=0;i<100;++i) CK(cuLaunchKernel(f,rows,1,1,128,1,1,512,stream,args,nullptr));
    CK(cudaStreamEndCapture(stream,&graph)); CK(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0));
    for(int i=0;i<3;++i) CK(cudaGraphLaunch(exec,stream));
    CK(cudaStreamSynchronize(stream));
    cudaEvent_t a,b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
    CK(cudaEventRecord(a,stream)); CK(cudaGraphLaunch(exec,stream)); CK(cudaEventRecord(b,stream)); CK(cudaEventSynchronize(b));
    float ms=0; CK(cudaEventElapsedTime(&ms,a,b));
    CK(cudaEventDestroy(a)); CK(cudaEventDestroy(b)); CK(cudaGraphExecDestroy(exec)); CK(cudaGraphDestroy(graph)); CK(cudaStreamDestroy(stream));
    return ms*10;
  }
};
int main(int argc,char **argv) {
  const bool production = argc==7 && strcmp(argv[6],"--production")==0;
  if(argc!=6 && !production) return 1;
  const int width=atoi(argv[3]), rows=atoi(argv[4]); const bool check=strcmp(argv[5],"check")==0;
  if(width<1 || width>2048 || rows<1 || rows>2048) return 1;
  CK(cudaSetDevice(0)); CK(cudaFree(nullptr));
  Buffer x(size_t(width)*rows*2), b(x.n), w(width*2), r(x.n), y(x.n), counts(16);
  int hc[]={rows>32?32:rows,0,rows>32?32:rows,rows};
  CK(cudaMemcpy(counts.p,hc,sizeof(hc),cudaMemcpyHostToDevice));
  Module old(argv[1],production), fresh(argv[2],production);
  for(int pattern=0;pattern<5;++pattern) {
    std::vector<__nv_bfloat16> hx(x.n/2),hb(x.n/2),hw(width);
    for(size_t i=0;i<hx.size();++i) {
      float v=float(int((i*17+pattern*11)%101)-50)/16.f;
      hx[i]=__float2bfloat16_rn(pattern==0?0.f:v);
      hb[i]=__float2bfloat16_rn(pattern==1?-__bfloat162float(hx[i]):float(int((i*13+7)%79)-39)/32.f);
    }
    for(int i=0;i<width;++i) hw[i]=__float2bfloat16_rn(float((i*7)%41+1)/16.f);
    w.put(hw);
    for(int alias=0;alias<2;++alias) {
      x.put(hx); b.put(hb);
      void *residual=alias?x.p:r.p;
      void *plain[]={&x.p,&b.p,&w.p,&residual,&y.p};
      void *framed[]={&counts.p,&x.p,&b.p,&w.p,&residual,&y.p};
      void **args=production?framed:plain;
      old.run(rows,args); CK(cudaDeviceSynchronize());
      auto expected=y.get(), er=alias?x.get():r.get();
      x.put(hx); b.put(hb); CK(cudaMemset(y.p,0x5a,y.n));
      fresh.run(rows,args); CK(cudaDeviceSynchronize());
      auto actual=y.get(), ar=alias?x.get():r.get();
      if(memcmp(actual.data(),expected.data(),y.n) || memcmp(er.data(),ar.data(),r.n)) { fprintf(stderr,"bitwise mismatch pattern=%d alias=%d\n",pattern,alias); return 3; }
      for(auto value:actual) if(!std::isfinite(__bfloat162float(value))) return 4;
    }
  }
  printf("correctness=passed bitwise=true width=%d rows=%d patterns=5 aliases=2\n",width,rows);
  if(!check) {
    void *plain[]={&x.p,&b.p,&w.p,&r.p,&y.p};
    void *framed[]={&counts.p,&x.p,&b.p,&w.p,&r.p,&y.p};
    void **args=production?framed:plain;
    for(int trial=0;trial<5;++trial) {
      float a,b;
      if(trial%2) { b=fresh.time(rows,args); a=old.time(rows,args); } else { a=old.time(rows,args); b=fresh.time(rows,args); }
      printf("trial=%d old_us=%.5f new_us=%.5f\n",trial,a,b);
    }
  }
}
