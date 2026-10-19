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
#else
static constexpr bool next_matrix=false;
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
static double datum(size_t i,int seed){return float(int((i*17+seed)%127)-63)/128.f;}
// Independent F64 mathematical attention with the declared BF16 probability
// boundary. This is a bounded analytic fixture, not framework agreement or
// whole-model quality. One BF16 rounding budget per weighted contribution plus
// output rounding supplies an absolute bound, including cancellation near zero.
static void wide_oracle(const std::vector<int>& selected,const std::vector<__nv_bfloat16>& output,int active) {
  for(int row:{0,active-1})for(int head:{0,31}) {
    std::vector<int> keys;std::vector<double> scores;
    for(int i=0;i<2051;++i)if(selected[row*2051+i]>=0)keys.push_back(selected[row*2051+i]);
    double mx=-INFINITY,sum=0;
    for(int key:keys){double dot=0;for(int c=0;c<256;++c)dot+=datum((row*32+head)*256+c,23)*datum((key*32+head)*256+c,29);scores.push_back(dot/16.);mx=std::max(mx,scores.back());}
    for(double s:scores)sum+=exp(s-mx);
    for(int c=0;c<256;++c){double y=0,mass=0;
      for(size_t i=0;i<keys.size();++i){double p=__bfloat162float(__float2bfloat16_rn(float(exp(scores[i]-mx)/sum)));double term=p*datum((keys[i]*32+head)*256+c,31);y+=term;mass+=fabs(term);}
      double actual=__bfloat162float(output[(row*32+head)*256+c]);
      const double bound=mass/128.+fabs(y)/256.+1e-6;
      if(fabs(actual-y)>bound){fprintf(stderr,"wide oracle failure row=%d head=%d c=%d actual=%.9g reference=%.9g bound=%.9g\n",row,head,c,actual,y,bound);exit(11);}
    }
  }
  puts("independent_f64_bf16_probability_bound=pass rows=2 heads=2 components=256");
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
    void *incremental_pool_args[]={&counts.p,&ik.p,&gates.p,&valid.p,&ape.p,&pooled.p,&raw.p,&pool_valid.p,&pool_count.p,&counts.p,&pos.p,&seq.p,&offset.p,&pages.p,&generations.p};
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
    auto launch=[&](int ver,int stage){
#ifdef next_SPLIT_INDEX
      if(ver==1&&stage==1)ck(cudaLaunchKernel((const void*)next_index_scores,dim3(next_grid_3),dim3(next_workers_3),index_args,0,nullptr));
#endif
      ck(cudaLaunchKernel(kernels[ver][stage],dim3(grid[ver][stage]),dim3(workers[ver][stage]),stage==0?(ver==0?pool_args:incremental_pool_args):stage==1?index_args:att_args,0,nullptr));};
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
          double squared=0,reference=0,maxabs=0;
          for(size_t i=0;i<output.size();++i){double a=__bfloat162float(output[i]),b=__bfloat162float(original_output[i]);squared+=(a-b)*(a-b);reference+=b*b;maxabs=std::max(maxabs,fabs(a-b));}
          printf("comparison history=%d active=%d index_differences=%zu output_differences=%zu relL2=%.9g maxabs=%.9g matrix=%d\n",history,active,di,dy,sqrt(squared/reference),maxabs,next_matrix);
          if(di||(!next_matrix&&dy))return 4;
          wide_oracle(indices,output,active);
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
