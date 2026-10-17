#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cstring>
#include <cmath>
#include "performance.cuh"
#ifndef LUNA_FP4_BANK_COUNT
#define LUNA_FP4_BANK_COUNT S
#endif
#ifdef LUNA_PAIRED
namespace candidate {
#include "candidate.cuh"
}
#endif
static void ck(cudaError_t e) {if(e!=cudaSuccess){fprintf(stderr,"%s\n",cudaGetErrorString(e));exit(2);}}
template<class T> static T *gpu(const std::vector<T>& v) {T *p;ck(cudaMalloc(&p,v.size()*sizeof(T)));ck(cudaMemcpy(p,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice));return p;}
int main(int argc,char **argv) {
  int rows=argc>1?atoi(argv[1]):1;
  bool trimmed=argc>2&&strcmp(argv[2],"trimmed")==0;
  bool once=false,diverse=false;
  for(int arg=3;arg<argc;arg++){once|=strcmp(argv[arg],"once")==0;diverse|=strcmp(argv[arg],"diverse")==0;}
  if(rows<1||rows>R) return 3;
  constexpr int banks=LUNA_FP4_BANK_COUNT;
  std::vector<unsigned char> bank(stride*banks);
  const int widths[3]={H,H,I},outs[3]={I,I,H};
  const size_t scales[3]={ws0,ws1,ws2},globals[3]={wg0,wg1,wg2};
  for(int e=0;e<banks;e++) for(int s=0;s<3;s++) {
    auto *p=bank.data()+e*stride+offsets[s];
    for(int r=0;r<outs[s];r++) {
      for(int k=0;k<widths[s];k+=2) {unsigned a=(r*3+k*5+e+s)&15,b=(r*3+(k+1)*5+e+s)&15;p[(r*widths[s]+k)/2]=a|(b<<4);}
      for(int k=0;k<widths[s]/16;k++) p[scales[s]+r*(widths[s]/16)+k]=32+(r+k+e+s)%5;
    }
    float global=.03125f;memcpy(p+globals[s],&global,4);
  }
  std::vector<__nv_bfloat16> x(R*H);for(int i=0;i<R*H;i++) x[i]=__float2bfloat16_rn(((i*7)%29-14)/32.f);
  std::vector<int> counts{0,0,0,rows,0},indices(R*S),map(288,-1);
  // A separate route-diversity control changes group occupancy, not weights:
  // 64 logical experts share eight synthetic banks. This does not reproduce
  // the real checkpoint's weight-cache footprint.
  int logical=diverse?(banks>S?banks:64):S;
  for(int j=0;j<R*S;j++) indices[j]=diverse?((j/S)*7+j%S)%logical:j%S;
  for(int e=0;e<logical;e++) map[e]=e%banks;
  auto *db=gpu(bank),*qx=gpu(std::vector<unsigned char>(qb0)),*qa=gpu(std::vector<unsigned char>(qb1));
  auto *dx=gpu(x),*a=gpu(std::vector<__nv_bfloat16>(activation_bytes/2));
  auto *c=gpu(counts),*idx=gpu(indices),*em=gpu(map),*err=gpu(std::vector<int>(1));
  auto *routes=gpu(std::vector<float>(R*S,.125f)),*w=gpu(std::vector<float>(R*S*H)),*out=gpu(std::vector<float>(R*H));
  // Host-trimmed grid is a diagnostic intervention with known synthetic routes,
  // not a proposed production CPU readback or dynamic launch mechanism.
  int group_counts[288]={};for(int j=0;j<rows*S;j++)group_counts[indices[j]]++;
  int groups=0;for(int e=0;e<288;e++)groups+=(group_counts[e]+15)/16;
  int gg=trimmed?groups*((I+31)/32):grid3,gd=trimmed?groups*((H+31)/32):grid6;
  printf("logical_experts=%d unique_weight_banks=%d live_groups=%d\n",logical,banks,groups);
  auto run=[&](bool fresh=false){
#ifdef LUNA_PAIRED
    if(fresh) {
      candidate::next_group<<<candidate::grid0,candidate::block0>>>(c,idx,a);
      candidate::next_quantize_0_global<<<grid1,block1>>>(c,dx,qx,err);
      candidate::next_quantize_0_pack<<<grid2,block2>>>(c,dx,qx,err);
      candidate::next_gate_up<<<trimmed?gg:candidate::grid3,candidate::block3>>>(c,qx,idx,em,db,a);
      candidate::next_quantize_1_global<<<grid4,block4>>>(c,a,qa,err);
      candidate::next_quantize_1_pack<<<grid5,block5>>>(c,a,qa,err);
      candidate::next_down<<<trimmed?gd:candidate::grid6,candidate::block6>>>(c,qa,idx,routes,em,db,w,a);
      candidate::next_combine<<<grid7,block7>>>(c,idx,w,out,routes);
      return;
    }
#endif
    perf_group<<<grid0,block0>>>(c,idx,a);
    perf_quantize_0_global<<<grid1,block1>>>(c,dx,qx,err);
    perf_quantize_0_pack<<<grid2,block2>>>(c,dx,qx,err);
    perf_gate_up<<<gg,block3>>>(c,qx,idx,em,db,a);
    perf_quantize_1_global<<<grid4,block4>>>(c,a,qa,err);
    perf_quantize_1_pack<<<grid5,block5>>>(c,a,qa,err);
    perf_down<<<gd,block6>>>(c,qa,idx,routes,em,db,w,a);
    perf_combine<<<grid7,block7>>>(c,idx,w,out,routes);
  };
  #ifdef LUNA_PAIRED
  auto snapshot=[&](){
    std::vector<unsigned char> bytes;
    auto add=[&](const void *p,size_t n){size_t start=bytes.size();bytes.resize(start+n);ck(cudaMemcpy(bytes.data()+start,p,n,cudaMemcpyDeviceToHost));};
    add(out,rows*H*4);add(w,rows*S*H*4);add(a,activation_bytes);add(qx,qb0);add(qa,qb1);
    return bytes;
  };
  run(false);ck(cudaDeviceSynchronize());auto reference=snapshot();
  run(true);ck(cudaDeviceSynchronize());auto actual=snapshot();
  if(reference!=actual){fprintf(stderr,"bitwise workspace/output mismatch\n");return 4;}
  printf("correctness=passed bitwise_chain=true bytes=%zu\n",actual.size());
  if(once){run(true);ck(cudaDeviceSynchronize());}
  else {
    cudaEvent_t start,end;ck(cudaEventCreate(&start));ck(cudaEventCreate(&end));
    auto time=[&](bool fresh){ck(cudaEventRecord(start));for(int n=0;n<20;n++)run(fresh);ck(cudaEventRecord(end));ck(cudaEventSynchronize(end));float ms;ck(cudaEventElapsedTime(&ms,start,end));return ms*1000/20;};
    for(int n=0;n<3;n++){run(false);run(true);}ck(cudaDeviceSynchronize());
    for(int trial=0;trial<5;trial++){float old_us,new_us;if(trial%2){new_us=time(true);old_us=time(false);}else{old_us=time(false);new_us=time(true);}printf("rows=%d trial=%d old_us=%.6f new_us=%.6f bitwise=true\n",rows,trial,old_us,new_us);}
    ck(cudaEventDestroy(start));ck(cudaEventDestroy(end));
  }
  #else
  if(once) {run();ck(cudaGetLastError());ck(cudaDeviceSynchronize());}
  else {for(int n=0;n<3;n++) run();ck(cudaDeviceSynchronize());cudaEvent_t start,end;ck(cudaEventCreate(&start));ck(cudaEventCreate(&end));
    for(int trial=0;trial<5;trial++){ck(cudaEventRecord(start));for(int n=0;n<20;n++)run();ck(cudaEventRecord(end));ck(cudaEventSynchronize(end));float ms;ck(cudaEventElapsedTime(&ms,start,end));printf("rows=%d trimmed=%d gate_ctas=%d down_ctas=%d trial=%d chain_us=%.6f\n",rows,trimmed,gg,gd,trial,ms*1000/20);}
    ck(cudaEventDestroy(start));ck(cudaEventDestroy(end));}
  #endif
  std::vector<float> host(rows*H);ck(cudaMemcpy(host.data(),out,host.size()*4,cudaMemcpyDeviceToHost));int nonfinite=0;for(float v:host)nonfinite+=!std::isfinite(v);
  int error;ck(cudaMemcpy(&error,err,4,cudaMemcpyDeviceToHost));printf("nonfinite=%d device_error=%d\n",nonfinite,error);
  ck(cudaFree(db));ck(cudaFree(qx));ck(cudaFree(qa));ck(cudaFree(dx));ck(cudaFree(a));ck(cudaFree(c));ck(cudaFree(idx));ck(cudaFree(em));ck(cudaFree(err));ck(cudaFree(routes));ck(cudaFree(w));ck(cudaFree(out));ck(cudaDeviceReset());
  return nonfinite||error?1:0;
}
