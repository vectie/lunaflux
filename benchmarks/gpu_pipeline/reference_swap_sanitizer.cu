// Standalone allocation/capture/replay/release test of the diagnostic shim.
// Real serving activation equivalence is checked separately by its shadow arm.
#include "reference_projection_swap.cu"
#include <dlfcn.h>

int main() {
  open_log();
  CHECK(cuInit(0));
  CUdevice device; CUcontext context;
  CHECK(cuDeviceGet(&device,0));
  CHECK(cuCtxCreate(&context,nullptr,0,device));
  original_destroy=&cuCtxDestroy;
  initialize();
  auto output=reinterpret_cast<CUfunction>(1);
  auto down=reinterpret_cast<CUfunction>(2);
  state.entries.push_back({output,1}); state.entries.push_back({down,2});
  std::vector<__nv_bfloat16> ones(size_t(M)*3072,__float2bfloat16(1.0f));
  void *x,*w,*y,*z,*count_ptr;
  CHECK(cudaMalloc(&x,size_t(M)*3072*2));
  CHECK(cudaMalloc(&w,size_t(N)*3072*2));
  CHECK(cudaMalloc(&y,size_t(M)*N*2));
  CHECK(cudaMalloc(&z,size_t(M)*N*2));
  CHECK(cudaMalloc(&count_ptr,4*sizeof(int)));
  CHECK(cudaMemcpy(x,ones.data(),size_t(M)*3072*2,cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(w,ones.data(),size_t(N)*3072*2,cudaMemcpyHostToDevice));
  int counts[4]={1,0,0,M};
  CHECK(cudaMemcpy(count_ptr,counts,sizeof(counts),cudaMemcpyHostToDevice));
  CUstream stream; CUgraph graph; CUgraphExec executable;
  CHECK(cuStreamCreate(&stream,CU_STREAM_NON_BLOCKING));
  CHECK(cuStreamBeginCapture(stream,CU_STREAM_CAPTURE_MODE_THREAD_LOCAL));
  void* output_args[]={&count_ptr,&x,&w,&y};
  void* down_args[]={&count_ptr,&x,&w,&w,&w,&z,&x};
  CHECK(lf_swap_launch(output,512,1,1,256,1,1,8192,stream,output_args,nullptr));
  CHECK(lf_swap_launch(down,256,1,1,128,1,1,0,stream,down_args,nullptr));
  CHECK(cuStreamEndCapture(stream,&graph));
  CHECK(cuGraphInstantiate(&executable,graph,0));
  for(int i=0;i<3;i++) CHECK(cuGraphLaunch(executable,stream));
  CHECK(cuStreamSynchronize(stream));
  std::vector<__nv_bfloat16> actual(size_t(M)*N);
  for(int i=0;i<2;i++) {
    CHECK(cudaMemcpy(actual.data(),i?z:y,size_t(M)*N*2,cudaMemcpyDeviceToHost));
    for(auto value:actual) if(__bfloat162float(value)!=(i?3072.f:2048.f)) std::abort();
  }
  CHECK(cuGraphExecDestroy(executable)); CHECK(cuGraphDestroy(graph));
  CHECK(cuStreamDestroy(stream));
  CHECK(cudaFree(x)); CHECK(cudaFree(w)); CHECK(cudaFree(y)); CHECK(cudaFree(z)); CHECK(cudaFree(count_ptr));
  CHECK(lf_swap_destroy(context));
  std::puts("graph_replays=3 checked_values=4194304 exact_constant_result=true explicit_release=true");
}
