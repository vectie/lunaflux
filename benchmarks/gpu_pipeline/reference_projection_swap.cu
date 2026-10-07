// Offline causal substitution, owned by the 2026-10-07 gap experiment.
// Never link into a release. Remove from an experimental worker after this trial.
// Only the pinned 2048-row output/down ABI and geometry are eligible. All other
// launches pass through. This is not a general runtime capability or promotion.
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <unistd.h>

#ifndef LF_SWAP_MASK
#define LF_SWAP_MASK 0
#endif
#ifndef LF_SWAP_VERIFY
#define LF_SWAP_VERIFY 0
#endif
#ifndef LF_SWAP_LOG_DIRECTORY
#error "A unique experiment log directory is required"
#endif
static FILE* diagnostic_log=nullptr;
static void open_log() {
  if(diagnostic_log) return;
  char path[4096];
  int n=std::snprintf(path,sizeof(path),"%s/worker-%ld.log",LF_SWAP_LOG_DIRECTORY,long(getpid()));
  if(n<0 || size_t(n)>=sizeof(path)) std::abort();
  // Approved workers deliberately close stderr. Diagnostic-only startup I/O,
  // exclusive creation, and explicit teardown reporting avoid changing that ABI.
  diagnostic_log=std::fopen(path,"wx");
  if(!diagnostic_log) std::abort();
  std::fprintf(diagnostic_log,"lf_swap started mask=%d verify=%d pid=%ld\n",LF_SWAP_MASK,LF_SWAP_VERIFY,long(getpid()));
  std::fflush(diagnostic_log);
}
using GetFunction = CUresult (*)(CUfunction*, CUmodule, const char*);
using Launch = CUresult (*)(CUfunction,unsigned,unsigned,unsigned,unsigned,unsigned,unsigned,unsigned,CUstream,void**,void**);
using Destroy = CUresult (*)(CUcontext);
static GetFunction original_get;
static Launch original_launch;
static Destroy original_destroy;
static constexpr int M=2048, N=1024;
static constexpr size_t workspace_bytes=32*1024*1024;
struct Entry { CUfunction function; int role; bool observed=false; };
struct State {
  CUcontext context=nullptr;
  cublasHandle_t blas=nullptr;
  void *workspace=nullptr, *scratch=nullptr;
  unsigned long long* checks=nullptr;
  std::vector<Entry> entries;
  unsigned long long substitutions[2]={0,0};
};
static thread_local State state;

static void checked(int code,const char* what) {
  if(code) { std::fprintf(diagnostic_log,"lf_swap failure=%s status=%d\n",what,code); std::fflush(diagnostic_log); std::abort(); }
}
#define CHECK(x) checked(int(x),#x)

static void gemm(State& s,int k,const void* x,const void* w,void* y,CUstream stream) {
  const float alpha=1.f,beta=0.f;
  CHECK(cublasSetStream(s.blas,reinterpret_cast<cudaStream_t>(stream)));
  // SetStream resets the workspace, so bind the same preallocated region again.
  CHECK(cublasSetWorkspace(s.blas,s.workspace,workspace_bytes));
  CHECK(cublasGemmEx(s.blas,CUBLAS_OP_T,CUBLAS_OP_N,N,M,k,&alpha,
    w,CUDA_R_16BF,k,x,CUDA_R_16BF,k,&beta,y,CUDA_R_16BF,N,
    CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

static void initialize() {
  CUcontext context;
  CHECK(cuCtxGetCurrent(&context));
  if(state.context) { if(state.context!=context) std::abort(); return; }
  state.context=context;
  CHECK(cublasCreate(&state.blas));
  CHECK(cublasSetMathMode(state.blas,CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION));
  CHECK(cudaMalloc(&state.workspace,workspace_bytes));
  CHECK(cudaMalloc(&state.scratch,size_t(M)*N*2));
  CHECK(cudaMalloc(&state.checks,4*sizeof(unsigned long long)));
  CHECK(cudaMemset(state.checks,0,4*sizeof(unsigned long long)));
  // Prime both fixed operations before stream capture. Vendor initialization
  // and selection stay outside timing; serving replays the captured kernels.
  void *x,*w;
  CHECK(cudaMalloc(&x,size_t(M)*3072*2));
  CHECK(cudaMalloc(&w,size_t(N)*3072*2));
  CHECK(cudaMemset(x,0,size_t(M)*3072*2));
  CHECK(cudaMemset(w,0,size_t(N)*3072*2));
  gemm(state,2048,x,w,state.scratch,nullptr);
  gemm(state,3072,x,w,state.scratch,nullptr);
  CHECK(cudaDeviceSynchronize());
  CHECK(cudaFree(x)); CHECK(cudaFree(w));
}

// Separate correctness run only. Compare actual live rows, never padded rows.
// BF16 input/output, F32 accumulation: reduction order may differ. A 2% relative
// + 0.01 absolute cross-backend gate is diagnostic, not a release accuracy claim.
__global__ void compare_live(const int* counts,const __nv_bfloat16* a,
    const __nv_bfloat16* b,unsigned long long* result) {
  const int live=counts[3];
  if(live<1 || live>M) { if(threadIdx.x==0 && blockIdx.x==0) atomicAdd(result+1,1ULL); return; }
  unsigned long long seen=0,bad=0,different=0; float maximum=0;
  for(int i=int(blockIdx.x*blockDim.x+threadIdx.x);i<live*N;i+=int(gridDim.x*blockDim.x)) {
    float x=__bfloat162float(a[i]),y=__bfloat162float(b[i]);
    float delta=fabsf(x-y),bound=0.01f+0.02f*fabsf(x);
    seen++;
    if(!isfinite(x)||!isfinite(y)||delta>bound) bad++;
    maximum=fmaxf(maximum,delta);
    if(__float_as_uint(x)!=__float_as_uint(y)) different++;
  }
  if(seen) atomicAdd(result,seen);
  if(bad) atomicAdd(result+1,bad);
  atomicMax(result+2,(unsigned long long)__float_as_uint(maximum));
  if(different) atomicAdd(result+3,different);
}

extern "C" CUresult lf_swap_get(CUfunction* out,CUmodule module,const char* name) {
  CUresult result=original_get(out,module,name);
  if(result!=CUDA_SUCCESS) return result;
  int role=std::strcmp(name,"lunaflux_luna_dense_projection_bf16_release_v1")==0?1:
    std::strcmp(name,"lunaflux_luna_gated_mlp_bf16_release_v1_down")==0?2:0;
  if(role) {
    std::fprintf(diagnostic_log,"lf_swap loaded=%s\n",name); std::fflush(diagnostic_log);
    initialize();
    for(auto& e:state.entries) if(e.function==*out) return result;
    state.entries.push_back({*out,role});
  }
  return result;
}

extern "C" CUresult lf_swap_launch(CUfunction f,unsigned gx,unsigned gy,unsigned gz,
    unsigned bx,unsigned by,unsigned bz,unsigned shared,CUstream stream,void** args,void** extra) {
  int role=0;
  for(auto& e:state.entries) if(e.function==f) {
    role=e.role;
    if(!e.observed) {
      std::fprintf(diagnostic_log,"lf_swap geometry role=%d grid=%u,%u,%u block=%u,%u,%u dynamic_shared=%u\n",role,gx,gy,gz,bx,by,bz,shared);
      std::fflush(diagnostic_log); e.observed=true;
    }
    break;
  }
  bool exact=gy==1&&gz==1&&by==1&&bz==1&&extra==nullptr&&args &&
    // Driver launch metadata is dynamic shared memory, not Nsight's total.
    // Output: 32768 static + 8192 dynamic; down: 49152 static + 0 dynamic.
    ((role==1&&gx==512&&bx==256&&shared==8192)||
     (role==2&&gx==256&&bx==128&&shared==0));
  if(!exact || !(LF_SWAP_MASK&role))
    return original_launch(f,gx,gy,gz,bx,by,bz,shared,stream,args,extra);
  const void* x=*static_cast<void**>(args[role==1?1:6]);
  const void* w=*static_cast<void**>(args[role==1?2:4]);
  void* y=*static_cast<void**>(args[role==1?3:5]);
  if(LF_SWAP_VERIFY) {
    auto result=original_launch(f,gx,gy,gz,bx,by,bz,shared,stream,args,extra);
    if(result!=CUDA_SUCCESS) return result;
    gemm(state,role==1?2048:3072,x,w,state.scratch,stream);
    compare_live<<<256,256,0,reinterpret_cast<cudaStream_t>(stream)>>>(
      *static_cast<int**>(args[0]),static_cast<__nv_bfloat16*>(y),
      static_cast<__nv_bfloat16*>(state.scratch),state.checks);
    CHECK(cudaGetLastError());
  } else {
    gemm(state,role==1?2048:3072,x,w,y,stream);
  }
  state.substitutions[role-1]++;
  return CUDA_SUCCESS;
}

extern "C" CUresult lf_swap_destroy(CUcontext context) {
  if(state.context==context) {
    CHECK(cuCtxSetCurrent(context)); CHECK(cudaDeviceSynchronize());
    unsigned long long checks[4]={};
    CHECK(cudaMemcpy(checks,state.checks,sizeof(checks),cudaMemcpyDeviceToHost));
    float maximum; unsigned int bits=unsigned(checks[2]); std::memcpy(&maximum,&bits,4);
    std::fprintf(diagnostic_log,"lf_swap mask=%d verify=%d captured_output=%llu captured_down=%llu checked=%llu violations=%llu max_abs=%g nonbitwise=%llu\n",
      LF_SWAP_MASK,LF_SWAP_VERIFY,state.substitutions[0],state.substitutions[1],checks[0],checks[1],maximum,checks[3]);
    std::fflush(diagnostic_log);
    CHECK(cublasDestroy(state.blas));
    CHECK(cudaFree(state.workspace)); CHECK(cudaFree(state.scratch)); CHECK(cudaFree(state.checks));
    state=State{};
    if(checks[1]) std::abort();
    std::fprintf(diagnostic_log,"lf_swap released=1\n");
    std::fflush(diagnostic_log);
  }
  return original_destroy(context);
}

extern "C" void* lf_reference_swap_symbol(const char* name,void* original) {
  open_log();
  if(std::strcmp(name,"cuModuleGetFunction")==0) {original_get=reinterpret_cast<GetFunction>(original);return reinterpret_cast<void*>(&lf_swap_get);}
  if(std::strcmp(name,"cuLaunchKernel")==0) {original_launch=reinterpret_cast<Launch>(original);return reinterpret_cast<void*>(&lf_swap_launch);}
  if(std::strcmp(name,"cuCtxDestroy_v2")==0) {original_destroy=reinterpret_cast<Destroy>(original);return reinterpret_cast<void*>(&lf_swap_destroy);}
  return original;
}
