// Offline actual-source replay. No dependency from production code.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "old.cuh"
#ifdef NEXT
#include "next.cuh"
#include "selection.cuh"
#endif
static void ck(cudaError_t e) {
  if(e!=cudaSuccess){fprintf(stderr,"CUDA: %s\n",cudaGetErrorString(e));exit(2);}
}
template<class T> struct D {
  T *p=nullptr; size_t n;
  explicit D(size_t n):n(n){ck(cudaMalloc(&p,n*sizeof(T)));}
  ~D(){ck(cudaFree(p));}
  void put(const std::vector<T>& v){if(v.size()!=n)exit(8);ck(cudaMemcpy(p,v.data(),n*sizeof(T),cudaMemcpyHostToDevice));}
  std::vector<T> get(){std::vector<T> v(n);ck(cudaMemcpy(v.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost));return v;}
};
static std::vector<__nv_bfloat16> data(size_t n,int seed) {
  std::vector<__nv_bfloat16> v(n);
  for(size_t i=0;i<n;++i)v[i]=__float2bfloat16_rn(float(int((i*17+seed)%127)-63)/128.f);
  return v;
}
int main(int argc,char**argv) {
  const char *mode=argc>1?argv[1]:"run";
  const bool profile=strstr(mode,"profile")!=nullptr;
  const bool sanitize=!strcmp(mode,"sanitize");
#ifdef NEXT
  if(!profile)for(int capacity:{16,1024})for(int pattern=0;pattern<4;++pattern) {
    int n=pattern==3?capacity-3:capacity;
    std::vector<float> scores(capacity);std::vector<int> flags(capacity),expected;
    for(int i=0;i<capacity;++i){
      scores[i]=pattern==0?0.f:float(i%13-6);
      flags[i]=pattern==2?0:(i%7!=3);
    }
    scores[0]=NAN;scores[1]=INFINITY;scores[2]=-INFINITY;scores[4]=-0.f;
    for(int i=0;i<n;++i)if(flags[i]&&!std::isnan(scores[i]))expected.push_back(i);
    std::sort(expected.begin(),expected.end(),[&](int a,int b){return scores[a]>scores[b]||(scores[a]==scores[b]&&a<b);});
    expected.resize(4,-1);
    D<float> s(capacity);D<int> a(capacity),y(4);s.put(scores);a.put(flags);
    void *args[]={&s.p,&a.p,&y.p,&n};
    ck(cudaLaunchKernel(capacity==16?(const void*)selection_edge_16:(const void*)selection_edge_1024,dim3(1),dim3(128),args,0,nullptr));
    ck(cudaDeviceSynchronize());if(y.get()!=expected)return 10;
    printf("selection_edge capacity=%d pattern=%d stable_exact=true\n",capacity,pattern);
  }
#endif
  {
    constexpr int rows=64,physical=8208,width=2051;
    D<int> counts(5),pos(rows),valid(physical),raw(physical),pool_valid(physical/4),pool_count(1),key_count(1);
    D<int> seq(rows),offset(2),pages(513),generations(513),selected(size_t(rows)*width);
    D<__nv_bfloat16> iq(rows*32*128),hw(rows*32),ik(physical*128),gates(physical*128),ape(4*128),pooled(physical/4*128);
    D<__nv_bfloat16> q(rows*32*256),k(size_t(physical)*32*256),v(size_t(physical)*32*256),out(rows*32*256);
    size_t bytes=old_scratch;
#ifdef NEXT
    bytes=std::max(bytes,size_t(next_scratch));
#endif
    D<unsigned char> scratch(bytes);
    iq.put(data(iq.n,3));hw.put(data(hw.n,7));ik.put(data(ik.n,11));gates.put(data(gates.n,13));ape.put(data(ape.n,19));
    q.put(data(q.n,23));k.put(data(k.n,29));v.put(data(v.n,31));
    seq.put(std::vector<int>(rows,0));offset.put({0,512});generations.put(std::vector<int>(513,1));
    std::vector<int> ids(513);for(int i=0;i<513;++i)ids[i]=i;pages.put(ids);
    void *pool_args[]={&counts.p,&ik.p,&gates.p,&valid.p,&ape.p,&pooled.p,&raw.p,&pool_valid.p,&pool_count.p};
    void *index_args[]={&counts.p,&pos.p,&iq.p,&hw.p,&pooled.p,&raw.p,&pool_valid.p,&pool_count.p,&valid.p,&selected.p,&scratch.p,&seq.p,&offset.p,&pages.p,&generations.p};
    void *att_args[]={&counts.p,&pos.p,&key_count.p,&valid.p,&q.p,&k.p,&v.p,&selected.p,&out.p,&scratch.p,&seq.p,&offset.p,&pages.p,&generations.p};
    const void *kernels[2][3]={{(const void*)old_pool,(const void*)old_index,(const void*)old_attention_retained_scores_v1}};
    int grid[2][3]={{old_grid_0,old_grid_1,old_grid_2}},workers[2][3]={{old_workers_0,old_workers_1,old_workers_2}};
    int versions=1;
#ifdef NEXT
    kernels[1][0]=(const void*)next_pool;kernels[1][1]=(const void*)next_index;kernels[1][2]=(const void*)next_attention_retained_scores_v1;
    grid[1][0]=next_grid_0;grid[1][1]=next_grid_1;grid[1][2]=next_grid_2;
    workers[1][0]=next_workers_0;workers[1][1]=next_workers_1;workers[1][2]=next_workers_2;versions=2;
#endif
    auto launch=[&](int ver,int stage){ck(cudaLaunchKernel(kernels[ver][stage],dim3(grid[ver][stage]),dim3(workers[ver][stage]),stage==0?pool_args:stage==1?index_args:att_args,0,nullptr));};
    for(int history: (profile||sanitize?std::vector<int>{8000}:std::vector<int>{64,256,2048,4096,8000,7997})) {
      int active=history==7997?29:64;
      counts.put({0,0,1,active,0});key_count.put({history});
      std::vector<int> p(rows,0),ok(physical,0);
      for(int i=0;i<active;++i)p[i]=history-active+i;
      for(int i=0;i<history;++i)ok[i]=1;
      pos.put(p);valid.put(ok);
      std::vector<int> original_indices;std::vector<__nv_bfloat16> original_output;
      for(int ver=0;ver<versions;++ver) {
        for(int stage=0;stage<3;++stage)launch(ver,stage);
        ck(cudaDeviceSynchronize());
        auto indices=selected.get();auto output=out.get();
        size_t nonzero=0,selected_count=0;
        for(auto x:output){if(!std::isfinite(__bfloat162float(x)))return 3;nonzero+=__bfloat162float(x)!=0.f;}
        for(auto x:indices)selected_count+=x>=0;
        if(!nonzero||selected_count<size_t(active))return 9;
        printf("executed history=%d version=%d selected=%zu nonzero_outputs=%zu\n",history,ver,selected_count,nonzero);
        if(ver==0){original_indices=indices;original_output=output;}
        else {
          size_t di=0,dy=0;
          for(size_t i=0;i<indices.size();++i)di+=indices[i]!=original_indices[i];
          for(size_t i=0;i<output.size();++i)dy+=memcmp(&output[i],&original_output[i],sizeof(output[i]))!=0;
          printf("correctness history=%d active=%d index_differences=%zu output_differences=%zu\n",history,active,di,dy);
          if(di||dy)return 4;
        }
      }
      if(profile||sanitize)continue;
      cudaEvent_t a,b;ck(cudaEventCreate(&a));ck(cudaEventCreate(&b));
      for(int trial=0;trial<3;++trial)for(int order=0;order<versions;++order){
        int ver=(order+trial)%versions;
        for(int stage=0;stage<3;++stage){
          // Index and attention share scratch; restore the required predecessor.
          launch(ver,0);launch(ver,1);
          ck(cudaEventRecord(a));for(int repeat=0;repeat<5;++repeat)launch(ver,stage);
          ck(cudaEventRecord(b));ck(cudaEventSynchronize(b));float ms=0;ck(cudaEventElapsedTime(&ms,a,b));
          printf("timing history=%d active=%d version=%d stage=%d trial=%d milliseconds=%.6f grid=%d workers=%d\n",history,active,ver,stage,trial,ms/5,grid[ver][stage],workers[ver][stage]);fflush(stdout);
        }
      }
      ck(cudaEventDestroy(a));ck(cudaEventDestroy(b));
    }
  }
  ck(cudaDeviceReset());puts("closed=true");
}
