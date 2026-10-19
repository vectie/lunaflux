// External mathematical fixtures for the actual generated kernel.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "source.cuh"
static void ck(cudaError_t e){if(e!=cudaSuccess){fprintf(stderr,"%s\n",cudaGetErrorString(e));exit(2);}}
template<class T> struct D {
  T *p;size_t n;
  D(size_t n):n(n){ck(cudaMalloc(&p,n*sizeof(T)));ck(cudaMemset(p,0,n*sizeof(T)));}
  ~D(){ck(cudaFree(p));}
  void put(const std::vector<T>& v){if(v.size()!=n)exit(3);ck(cudaMemcpy(p,v.data(),n*sizeof(T),cudaMemcpyHostToDevice));}
  std::vector<T> get(){std::vector<T> v(n);ck(cudaMemcpy(v.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost));return v;}
};
using B=__nv_bfloat16;
static std::vector<B> data(size_t n,int salt){std::vector<B> a(n);for(size_t i=0;i<n;++i)a[i]=__float2bfloat16_rn(float(int((i*17+salt)%127)-63)/64.f);return a;}
int main(){
  {
    D<int> counts(5),positions(31),key_count(1),valid(96),selection(31*17),sequences(31),offsets(3),pages(16),generations(16);
    D<B> q(31*4*32),k(96*2*32),v(96*2*24),out(31*4*24);D<float> scratch(31*4*64);
    auto qh=data(q.n,3),kh=data(k.n,11),vh=data(v.n,27);q.put(qh);k.put(kh);v.put(vh);key_count.put({64});
    std::vector<int> pg={5,1,9,4,6,2,8,0,0,0,0,0,0,0,0,0},gen(16,0);
    for(int i=0;i<8;++i)gen[i]=i<4?1:2;
    offsets.put({0,4,8});pages.put(pg);generations.put(gen);
    void *args[]={&counts.p,&positions.p,&key_count.p,&valid.p,&q.p,&k.p,&v.p,&selection.p,&out.p,&scratch.p,&sequences.p,&offsets.p,&pages.p,&generations.p};
    for(int mode=0;mode<12;++mode){
      int active=mode==1?17:mode==2?1:mode==3?0:mode==9?15:mode==10?16:31,nseq=mode==6?3:2;
      std::vector<int> pos(31),seq(31),selected(31*17,-1),ok(96,0);
      for(int i=0;i<8;++i)for(int c=0;c<8;++c)ok[pg[i]*8+c]=gen[i];
      for(int r=0;r<31;++r){
        pos[r]=8+r%24;seq[r]=r%2;
        for(int rank=0;rank<std::min(17,pos[r]+1);++rank)selected[r*17+rank]=pos[r]-rank;
      }
      if(mode==4){selected[7*17+1]=selected[7*17];selected[9*17]=-2;selected[3*17]=pos[3]+1;}
      if(mode==5)for(int c=0;c<8;++c)ok[pg[1]*8+c]=7;
      if(mode==7)seq[5]=2;
      if(mode==11)std::fill(selected.begin(),selected.end(),-1);
      counts.put({0,0,nseq,mode==8?32:active,0});positions.put(pos);sequences.put(seq);selection.put(selected);valid.put(ok);
      std::vector<B> first;
      for(int repeat=0;repeat<3;++repeat){
        ck(cudaLaunchKernel((const void*)reuse_fixture,dim3(124),dim3(256),args,0,nullptr));ck(cudaDeviceSynchronize());
        auto actual=out.get();
        if(repeat&&memcmp(first.data(),actual.data(),actual.size()*sizeof(B))){fprintf(stderr,"nondeterministic mode=%d\n",mode);return 4;}
        first=actual;
      }
      for(int r=0;r<31;++r)for(int head=0;head<4;++head){
        bool row_ok=r<active&&mode!=6&&mode!=8&&seq[r]>=0&&seq[r]<2;
        std::vector<int> retained;std::vector<bool> seen(64,false);
        for(int rank=0;rank<17&&row_ok;++rank){
          int key=selected[r*17+rank];if(key==-1)continue;
          if(key<0||key>pos[r]||seen[key]){row_ok=false;break;}
          seen[key]=true;int at=seq[r]*4+key/8,physical=pg[at]*8+key%8;
          if(ok[physical]!=gen[at]){row_ok=false;break;}retained.push_back(physical);
        }
        std::vector<double> logits;double maximum=-INFINITY,denominator=0;
        if(row_ok)for(int raw:retained){
          double dot=0;for(int c=0;c<32;++c)dot+=double(__bfloat162float(qh[(r*4+head)*32+c]))*__bfloat162float(kh[(raw*2+head/2)*32+c]);
          logits.push_back(dot/sqrt(32.));maximum=std::max(maximum,logits.back());
        }
        for(double x:logits)denominator+=exp(x-maximum);
        for(int c=0;c<24;++c){
          double result=0,mass=0;
          if(row_ok)for(size_t i=0;i<retained.size();++i){
            double p=__bfloat162float(__float2bfloat16_rn(float(exp(logits[i]-maximum)/denominator)));
            double term=p*__bfloat162float(vh[(retained[i]*2+head/2)*24+c]);result+=term;mass+=fabs(term);
          }
          double actual=__bfloat162float(first[(r*4+head)*24+c]),bound=mass/128.+fabs(result)/256.+1e-6;
          if(!std::isfinite(actual)||fabs(actual-result)>bound||(!row_ok&&actual!=0)){fprintf(stderr,"oracle mode=%d row=%d head=%d c=%d actual=%g reference=%g bound=%g\n",mode,r,head,c,actual,result,bound);return 5;}
        }
      }
      printf("mode=%d active=%d all_rows_all_heads_f64=pass deterministic_repeats=3\n",mode,active);
    }
  }
  ck(cudaDeviceReset());puts("released=true");
}
