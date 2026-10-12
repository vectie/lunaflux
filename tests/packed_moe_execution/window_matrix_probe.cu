// Offline numerical candidate. No serving selection or alternate reference.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include "window_matrix_kernels.cuh"

static void check(cudaError_t error) {
  if(error!=cudaSuccess){std::fprintf(stderr,"%s\n",cudaGetErrorString(error));std::exit(2);}
}
template<class T> static T *allocate(size_t count) {
  T *result=nullptr;check(cudaMalloc(&result,count*sizeof(T)));
  check(cudaMemset(result,0,count*sizeof(T)));return result;
}
template<class T> static std::vector<T> read(const std::string &path,size_t count) {
  auto *file=std::fopen(path.c_str(),"rb");if(!file)std::exit(3);
  std::vector<T> values(count);
  const bool exact=std::fread(values.data(),sizeof(T),count,file)==count&&std::fgetc(file)==EOF;
  const bool closed=std::fclose(file)==0;if(!exact||!closed)std::exit(3);return values;
}
template<class T> static T *upload(const std::vector<T> &values) {
  auto *result=allocate<T>(values.size());
  check(cudaMemcpy(result,values.data(),values.size()*sizeof(T),cudaMemcpyHostToDevice));return result;
}
static std::vector<uint16_t> capture(__nv_bfloat16 *value,size_t count,const std::string &path) {
  std::vector<uint16_t> result(count);
  check(cudaMemcpy(result.data(),value,count*2,cudaMemcpyDeviceToHost));
  auto *file=std::fopen(path.c_str(),"wbx");if(!file)std::exit(4);
  const bool written=std::fwrite(result.data(),2,count,file)==count;
  const bool closed=std::fclose(file)==0;if(!written||!closed)std::exit(4);return result;
}

int main(int argc,char **argv) {
  if(argc!=3)return 2;
  const std::string input(argv[1]),output(argv[2]);
  const auto query=read<__nv_bfloat16>(input+"/q-rope.bf16",144*64*512);
  const auto keys=read<__nv_bfloat16>(input+"/kv-sim.bf16",144*512);
  const auto sinks=read<float>(input+"/sink.f32",64);
  auto *q=upload(query),*kv=upload(keys);
  auto *sink=upload(sinks);
  auto *ring=allocate<__nv_bfloat16>(128*512),*out=allocate<__nv_bfloat16>(144*64*512);
  auto *append=allocate<int>(3);int counts[3]={0,144,0};
  check(cudaMemcpy(append,counts,sizeof(counts),cudaMemcpyHostToDevice));
  matched_window_attention<<<dim3(144,MATRIX_GRID_Y),MATRIX_THREADS>>>(append,q,kv,ring,sink,out);
  check(cudaGetLastError());check(cudaDeviceSynchronize());
  const auto prefill=capture(out,144*64*512,output+"/prefill.bf16");
  for(int position=0;position<144;++position) {
    counts[0]=position;counts[1]=1;
    check(cudaMemcpy(append,counts,sizeof(counts),cudaMemcpyHostToDevice));
    matched_window_attention<<<dim3(1,MATRIX_GRID_Y),MATRIX_THREADS>>>(append,q+position*64*512,kv+position*512,ring,sink,out+position*64*512);
    matched_window_copy<<<128,128>>>(append,reinterpret_cast<const uint16_t*>(kv+position*512),reinterpret_cast<uint16_t*>(ring));
    check(cudaGetLastError());
  }
  check(cudaDeviceSynchronize());
  const auto incremental=capture(out,144*64*512,output+"/incremental.bf16");
  const bool matched=prefill==incremental;
  // Head-group and output-column tails on genuine BF16 input values. This is
  // an independent shape control, not another original-model execution.
  std::vector<__nv_bfloat16> tq(144*18*80),tk(144*80);
  std::vector<float> ts(sinks.begin(),sinks.begin()+18);
  for(int row=0;row<144;++row) {
    for(int head=0;head<18;++head)for(int col=0;col<80;++col)tq[(row*18+head)*80+col]=query[(row*64+head)*512+col];
    for(int col=0;col<80;++col)tk[row*80+col]=keys[row*512+col];
  }
  auto *tquery=upload(tq),*tkeys=upload(tk);
  auto *tsink=upload(ts);
  auto *tring=allocate<__nv_bfloat16>(128*80),*tout=allocate<__nv_bfloat16>(144*18*80);
  counts[0]=0;counts[1]=144;
  check(cudaMemcpy(append,counts,sizeof(counts),cudaMemcpyHostToDevice));
  tail_window_attention<<<dim3(144,TAIL_GRID_Y),TAIL_THREADS>>>(append,tquery,tkeys,tring,tsink,tout);
  check(cudaGetLastError());check(cudaDeviceSynchronize());
  const auto tail_prefill=capture(tout,144*18*80,output+"/tail-prefill.bf16");
  for(int position=0;position<144;++position) {
    counts[0]=position;counts[1]=1;
    check(cudaMemcpy(append,counts,sizeof(counts),cudaMemcpyHostToDevice));
    tail_window_attention<<<dim3(1,TAIL_GRID_Y),TAIL_THREADS>>>(append,tquery+position*18*80,tkeys+position*80,tring,tsink,tout+position*18*80);
    tail_window_copy<<<128,128>>>(append,reinterpret_cast<const uint16_t*>(tkeys+position*80),reinterpret_cast<uint16_t*>(tring));
    check(cudaGetLastError());
  }
  check(cudaDeviceSynchronize());
  const auto tail_incremental=capture(tout,144*18*80,output+"/tail-incremental.bf16");
  const bool tail_matched=tail_prefill==tail_incremental;
  for(void *pointer:{(void*)q,(void*)kv,(void*)sink,(void*)ring,(void*)out,(void*)append,
      (void*)tquery,(void*)tkeys,(void*)tsink,(void*)tring,(void*)tout})check(cudaFree(pointer));
  if(!matched||!tail_matched){std::fprintf(stderr,"submission geometry changed numerical output\n");return 5;}
  std::puts("matrix_window_probe completed=true prefill_incremental_bitwise=true head_column_tails_bitwise=true release=balanced");
}
