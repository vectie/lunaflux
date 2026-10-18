#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
namespace old {
#include "baseline.cuh"
}
#undef LUNA_FP4_BANK_COUNT
namespace next {
using old::lf_precision_e4;
using old::lf_precision_encode_e2;
#include "candidate.cuh"
}
static void ck(cudaError_t e) {
  if(e!=cudaSuccess){fprintf(stderr,"%s\n",cudaGetErrorString(e));exit(2);}
}
template<class T> static T *alloc(size_t n) {
  T *p;ck(cudaMalloc(&p,n*sizeof(T)));return p;
}
static void load(const std::string &path,void *device,size_t bytes) {
  FILE *f=fopen(path.c_str(),"rb");if(!f){perror(path.c_str());exit(3);}
  std::vector<unsigned char> data(bytes);
  if(fread(data.data(),1,bytes,f)!=bytes||fgetc(f)!=EOF){fprintf(stderr,"geometry: %s\n",path.c_str());exit(3);}
  fclose(f);ck(cudaMemcpy(device,data.data(),bytes,cudaMemcpyHostToDevice));
}
// Actual compiler-generated down kernel and literal retired operand storage.
// Optional literal TP-sharded down weights replace the synthetic control.
// Neither mode substitutes for complete-model or NCCL acceptance.
int main(int argc,char **argv) {
  if(argc!=5&&argc!=6){fprintf(stderr,"PREFIX LIVE_ROWS ITERATIONS serial|concurrent [LITERAL_BANK_DIR]\n");return 2;}
  constexpr int R=next::R,H=next::H,S=next::S,I=next::I,banks=LUNA_FP4_BANK_COUNT;
  const bool literal=argc==6;
  int live=atoi(argv[2]),iterations=atoi(argv[3]);
  bool concurrent=strcmp(argv[4],"concurrent")==0;
  if(live<=0||live>R||iterations<=0||iterations>10000||
     (!concurrent&&strcmp(argv[4],"serial")!=0))return 2;
  auto counts=alloc<int>(5),indices=alloc<int>(R*S),map=alloc<int>(288);
  auto weights=alloc<float>(R*S),output=alloc<float>(R*S*H),other=alloc<float>(R*S*H);
  auto activation=alloc<unsigned char>(next::activation_bytes);
  auto qa=alloc<unsigned char>(next::physical_qb1);
  auto bank=alloc<unsigned char>(next::stride*banks),raw=alloc<unsigned char>(next::stride);
  auto temporary=alloc<unsigned char>(next::stride);
  std::string prefix=argv[1];
  load(prefix+"routed-activation.bin",activation,next::activation_bytes);
  load(prefix+"routed-quantized-activation.bin",qa,next::physical_qb1);
  load(prefix+"router-indices.bin",indices,R*S*4);
  load(prefix+"router-weights.bin",weights,R*S*4);
  int count_values[5]={0,0,0,live,0};
  ck(cudaMemcpy(counts,count_values,sizeof(count_values),cudaMemcpyHostToDevice));
  std::vector<int> mapping(288,-1),observed_indices(R*S);
  ck(cudaMemcpy(observed_indices.data(),indices,R*S*4,cudaMemcpyDeviceToHost));
  if(literal) {
    if(banks!=288)return 2;
    for(int job=0;job<live*S;job++) {
      int e=observed_indices[job];if(e<0||e>=288)return 3;mapping[e]=e;
    }
  } else for(int e=0;e<288;e++)mapping[e]=e%banks;
  ck(cudaMemcpy(map,mapping.data(),mapping.size()*4,cudaMemcpyHostToDevice));
  const size_t scales[3]={next::ws0,next::ws1,next::ws2},globals[3]={next::wg0,next::wg1,next::wg2};
  const int inner[3]={H,H,I},columns[3]={I,I,H};
  std::vector<unsigned char> data(next::stride);
  unsigned state=7;
  ck(cudaMemset(bank,0,next::stride*banks));
  for(int e=0;e<banks;e++) {
    if(literal&&mapping[e]<0)continue;
    std::fill(data.begin(),data.end(),0);
    for(int s=0;!literal&&s<3;s++) {
      auto p=data.data()+next::offsets[s];
      for(int i=0;i<inner[s]*columns[s]/2;i++){state=state*1664525U+1013904223U;p[i]=state>>24;}
      for(int i=0;i<inner[s]*columns[s]/16;i++){state=state*1664525U+1013904223U;p[scales[s]+i]=32+(state>>24)%16;}
      float global=.03125f;memcpy(p+globals[s],&global,4);
    }
    ck(cudaMemcpy(raw,data.data(),data.size(),cudaMemcpyHostToDevice));
    if(literal) {
      const std::string base=std::string(argv[5])+"/expert-"+std::to_string(e);
      load(base+".payload",raw+next::offsets[2],size_t(H)*I/2);
      load(base+".scales",raw+next::offsets[2]+next::ws2,size_t(H)*I/16);
      load(base+".global",raw+next::offsets[2]+next::wg2,4);
    }
    next::next_prepare_bank<<<256,256>>>(raw,temporary);
    next::next_prepare_bank_copy<<<256,256>>>(temporary,bank+e*next::stride);
    ck(cudaDeviceSynchronize());
  }
  ck(cudaFree(raw));ck(cudaFree(temporary));
  cudaStream_t stream,aux;ck(cudaStreamCreate(&stream));ck(cudaStreamCreate(&aux));
  auto down=[&](float *out,cudaStream_t lane){
    next::next_down<<<next::grid8,next::block8,0,lane>>>(counts,qa,indices,weights,map,bank,out,reinterpret_cast<const __nv_bfloat16*>(activation));
    ck(cudaGetLastError());
  };
  size_t elements=size_t(live)*S*H;
  std::vector<float> reference(elements),observed(elements),parallel(elements);
  down(output,stream);ck(cudaStreamSynchronize(stream));
  ck(cudaMemcpy(reference.data(),output,elements*4,cudaMemcpyDeviceToHost));
  size_t different=0,nonfinite=0;
  for(float v:reference)if(!std::isfinite(v))nonfinite++;
  if(literal&&iterations>2) {
    const std::string path=std::string(argv[5])+"/partial-"+argv[4]+".f32";
    FILE *f=fopen(path.c_str(),"wbx");if(!f){perror(path.c_str());return 3;}
    if(fwrite(reference.data(),4,elements,f)!=elements||fclose(f))return 3;
  }
  for(int step=0;step<iterations;step++) {
    down(other,concurrent?aux:stream);down(output,stream);
    ck(cudaStreamSynchronize(stream));ck(cudaStreamSynchronize(aux));
    ck(cudaMemcpy(observed.data(),output,elements*4,cudaMemcpyDeviceToHost));
    ck(cudaMemcpy(parallel.data(),other,elements*4,cudaMemcpyDeviceToHost));
    for(size_t i=0;i<elements;i++)if(memcmp(&observed[i],&reference[i],4)||memcmp(&parallel[i],&reference[i],4)) {
      if(different<16)printf("step=%d index=%zu serial=%.9g primary=%.9g auxiliary=%.9g\n",step,i,reference[i],observed[i],parallel[i]);
      different++;
    }
    if(different)break;
  }
  printf("literal_rows=true synthetic_weights=%s tp=false concurrent=%s iterations=%d differing_f32=%zu nonfinite=%zu\n",literal?"false":"true",concurrent?"true":"false",iterations,different,nonfinite);
  ck(cudaStreamDestroy(stream));ck(cudaStreamDestroy(aux));
  void *owned[]={counts,indices,map,weights,output,other,activation,qa,bank};
  for(void *p:owned)ck(cudaFree(p));
  ck(cudaDeviceReset());puts("released=true");return different||nonfinite?4:0;
}
