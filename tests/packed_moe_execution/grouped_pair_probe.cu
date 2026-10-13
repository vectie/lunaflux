#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "baseline.cuh"
#include "candidate.cuh"
static void ck(cudaError_t e){if(e!=cudaSuccess){fprintf(stderr,"%s\n",cudaGetErrorString(e));exit(1);}}
template<class T>static T* alloc(size_t n){T*p;ck(cudaMalloc(&p,n*sizeof(T)));return p;}
int main(int argc,char**argv){
 const int H=4096,I=2048,R=16,S=8,J=R*S,E=32;
 const int rows=argc>1?atoi(argv[1]):16;const char*mode=argc>2?argv[2]:"timing";
 if(rows<1||rows>R)return 2;
 static_assert(baseline_stride==candidate_stride);
 auto bank=alloc<unsigned char>(candidate_stride*E);
 for(int e=0;e<E;++e)for(int stage=0;stage<3;++stage){
   size_t elements=H*I;auto p=bank+e*candidate_stride+candidate_offsets[stage];
   // Include both signs and nonuniform packed values/scales, not just the
   // preceding all-constant dyadic throughput fixture.
   std::vector<unsigned char> payload(elements/2),scales(elements/16);
   unsigned rng=1701+e*13+stage*19;
   for(auto&v:payload){rng=rng*1664525U+1013904223U;v=(unsigned char)(rng>>24);}
   for(size_t i=0;i<scales.size();++i)scales[i]=(unsigned char)(0x30+(i%9));
   ck(cudaMemcpy(p,payload.data(),payload.size(),cudaMemcpyHostToDevice));
   ck(cudaMemcpy(p+elements/2,scales.data(),scales.size(),cudaMemcpyHostToDevice));
   float scale=.017f;ck(cudaMemcpy(p+elements/2+elements/16,&scale,4,cudaMemcpyHostToDevice));
 }
 auto x=alloc<__nv_bfloat16>(R*H);std::vector<__nv_bfloat16>hx(R*H);
 for(int i=0;i<R*H;++i)hx[i]=__float2bfloat16_rn((i%31-15)*.013f);
 ck(cudaMemcpy(x,hx.data(),R*H*2,cudaMemcpyHostToDevice));
 auto counts=alloc<int>(5),indices=alloc<int>(J),map=alloc<int>(288);auto weights=alloc<float>(J);
 int hc[]={0,1,1,rows,0};std::vector<int>hi(J),hm(288,-1);std::vector<float>hw(J);
 for(int i=0;i<E;++i)hm[i]=i;for(int r=0;r<R;++r)for(int s=0;s<S;++s){hi[r*S+s]=(r*3+s)%E;hw[r*S+s]=(s+1)/36.f;}
 ck(cudaMemcpy(counts,hc,20,cudaMemcpyHostToDevice));ck(cudaMemcpy(indices,hi.data(),J*4,cudaMemcpyHostToDevice));ck(cudaMemcpy(map,hm.data(),288*4,cudaMemcpyHostToDevice));ck(cudaMemcpy(weights,hw.data(),J*4,cudaMemcpyHostToDevice));
 auto a=alloc<unsigned char>(candidate_activation);auto b=alloc<__nv_bfloat16>(J*H);auto y=alloc<float>(R*H);auto ab=(__nv_bfloat16*)a;
 auto run=[&](bool fresh){
  if(fresh){candidate_group<<<candidate_grid0,candidate_block0>>>(counts,indices,ab);candidate_gate_up<<<candidate_grid1,candidate_block1>>>(counts,x,indices,map,bank,ab);candidate_down<<<candidate_grid2,candidate_block2>>>(counts,ab,indices,weights,map,bank,b);candidate_combine<<<candidate_grid3,candidate_block3>>>(counts,indices,b,y);}
  else{baseline_group<<<baseline_grid0,baseline_block0>>>(counts,indices,ab);baseline_gate_up<<<baseline_grid1,baseline_block1>>>(counts,x,indices,map,bank,ab);baseline_down<<<baseline_grid2,baseline_block2>>>(counts,ab,indices,weights,map,bank,b);baseline_combine<<<baseline_grid3,baseline_block3>>>(counts,indices,b,y);}
  ck(cudaGetLastError());
 };
 if(!strcmp(mode,"candidate")||!strcmp(mode,"baseline")){run(!strcmp(mode,"candidate"));ck(cudaDeviceSynchronize());}
 else{
  std::vector<float>ref(rows*H),actual(rows*H);std::vector<unsigned char>aref(rows*S*I*2);std::vector<__nv_bfloat16>bref(rows*S*H);
  run(false);ck(cudaMemcpy(ref.data(),y,rows*H*4,cudaMemcpyDeviceToHost));ck(cudaMemcpy(aref.data(),a,aref.size(),cudaMemcpyDeviceToHost));ck(cudaMemcpy(bref.data(),b,bref.size()*2,cudaMemcpyDeviceToHost));
  run(true);ck(cudaMemcpy(actual.data(),y,rows*H*4,cudaMemcpyDeviceToHost));
  std::vector<unsigned char>anew(aref.size());std::vector<__nv_bfloat16>bnew(bref.size());ck(cudaMemcpy(anew.data(),a,anew.size(),cudaMemcpyDeviceToHost));ck(cudaMemcpy(bnew.data(),b,bnew.size()*2,cudaMemcpyDeviceToHost));
  if(memcmp(ref.data(),actual.data(),rows*H*4)||memcmp(aref.data(),anew.data(),anew.size())||memcmp(bref.data(),bnew.data(),bnew.size()*2)){fprintf(stderr,"output/intermediate mismatch\n");return 3;}
  puts("correctness=passed bitwise_chain=true");
  for(int trial=0;trial<5;++trial){float times[2];for(int order=0;order<2;++order){bool fresh=((trial+order)&1)!=0;cudaEvent_t begin,end;ck(cudaEventCreate(&begin));ck(cudaEventCreate(&end));ck(cudaEventRecord(begin));for(int i=0;i<10;++i)run(fresh);ck(cudaEventRecord(end));ck(cudaEventSynchronize(end));float ms;ck(cudaEventElapsedTime(&ms,begin,end));times[fresh]=ms*100;ck(cudaEventDestroy(begin));ck(cudaEventDestroy(end));}printf("rows=%d trial=%d old_us=%.9f new_us=%.9f\n",rows,trial,times[0],times[1]);}
 }
 for(void*p:{(void*)bank,(void*)x,(void*)counts,(void*)indices,(void*)map,(void*)weights,(void*)a,(void*)b,(void*)y})ck(cudaFree(p));ck(cudaDeviceReset());puts("released=true");
}
