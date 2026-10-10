// Independent small CPU oracle: previous-token dependencies, raw confidence,
// stable greedy ties, explicit stochastic replay, inactive and invalid rows.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "markov.cuh"
static void check(cudaError_t e) { if(e!=cudaSuccess) {std::fprintf(stderr,"%s\n",cudaGetErrorString(e));std::exit(2);} }
template<class T> static T* alloc(size_t n) {T* p=nullptr;check(cudaMallocManaged(&p,n*sizeof(T)));return p;}
static float bf(__nv_bfloat16 x) {return __bfloat162float(x);}
int main() {
  constexpr int V=17,H=8,K=4,R=5;
  auto count=alloc<int>(1), row=alloc<int>(R), tokens=alloc<int>(R+1);
  auto rng=alloc<uint64_t>(2);
  auto hidden=alloc<__nv_bfloat16>(R*H), table=alloc<__nv_bfloat16>(V*K);
  auto weight=alloc<__nv_bfloat16>(V*K), cw=alloc<__nv_bfloat16>(H+K), embeds=alloc<__nv_bfloat16>(R*K);
  auto logits=alloc<float>(R*V), confidence=alloc<float>(R);
  for(int i=0;i<R;++i) row[i]=i;
  for(int i=0;i<R*H;++i) hidden[i]=__float2bfloat16_rn(float(i%13-6)*.25f);
  for(int i=0;i<V*K;++i) {
    table[i]=__float2bfloat16_rn(float((i*7)%19-9)*.25f);
    weight[i]=__float2bfloat16_rn(float((i*11)%17-8)*.125f);
  }
  for(int i=0;i<H+K;++i) cw[i]=__float2bfloat16_rn(float(i%7-3)*.5f);
  std::vector<float> initial(R*V); for(int i=0;i<R*V;++i) initial[i]=float((i*5)%23-11)*.03125f;
  auto initialize=[&](int n,int seed,uint64_t sequence) {
    count[0]=n; tokens[0]=seed;
    for(int i=1;i<=R;++i) tokens[i]=-99;
    std::copy(initial.begin(),initial.end(),logits);
    std::fill(confidence,confidence+R,-123.f);
    rng[0]=147ULL;rng[1]=sequence;
  };
  auto submit=[&](bool random) {
    for(int i=0;i<R;++i) {
      if(random) {
        chain_random_gather<<<1,256>>>(count,tokens,table,embeds,row+i);
        chain_random_bias<<<1,256>>>(count,tokens,embeds,weight,logits,row+i);
        chain_random_sample<<<1,256>>>(count,logits,rng,tokens,row+i);
        chain_random_confidence<<<1,1>>>(count,tokens,hidden,embeds,cw,confidence,row+i);
      } else {
        chain_greedy_gather<<<1,256>>>(count,tokens,table,embeds,row+i);
        chain_greedy_bias<<<1,256>>>(count,tokens,embeds,weight,logits,row+i);
        chain_greedy_sample<<<1,256>>>(count,logits,rng,tokens,row+i);
        chain_greedy_confidence<<<1,1>>>(count,tokens,hidden,embeds,cw,confidence,row+i);
      }
    }
    check(cudaGetLastError());check(cudaDeviceSynchronize());
  };
  int cases=0,changed=0;float worst=0;
  auto main_counts=alloc<int>(5),start=alloc<int>(1),seed=alloc<int>(1);
  auto draft_counts=alloc<int>(5),ids=alloc<int>(R),positions=alloc<int>(R);
  for(int n : {0,1,8,9}) for(int base : {-1,0,31,2147483647}) for(int token : {-1,0,16,17}) {
    std::fill(main_counts,main_counts+5,0);main_counts[3]=n;start[0]=base;seed[0]=token;
    std::fill(tokens,tokens+R+1,-99);
    seed_fixture_seed<<<1,1>>>(main_counts,start,seed,draft_counts,ids,positions,tokens);
    check(cudaGetLastError());check(cudaDeviceSynchronize());
    bool valid=n>0 && n<=8 && base>=0 && base<=2147483647-n-R && token>=0 && token<V;
    if(draft_counts[2]!=1 || draft_counts[3]!=(valid?R:0) || tokens[0]!=(valid?token:-1)) return 9;
    for(int i=0;i<R;++i) {
      if(ids[i]!=(i==0?token:16) || positions[i]!=(valid?base+n+i:0) || tokens[i+1]!=-99) return 10;
    }
    ++cases;
  }
  auto compare=[&](int n,int seed,bool random) {
    int previous=seed;
    for(int i=0;i<R;++i) {
      if(n<=0 || n>R || i>=n) {
        if(tokens[i+1]!=-99 || confidence[i]!=-123.f) return false;
        for(int w=0;w<V;++w) if(logits[i*V+w]!=initial[i*V+w]) return false;
        continue;
      }
      if(previous<0 || previous>=V) {
        if(tokens[i+1]!=-1 || !std::isnan(confidence[i])) return false;
        for(int w=0;w<V;++w) if(!std::isnan(logits[i*V+w])) return false;
        previous=-1;continue;
      }
      int best=0;float best_value=-INFINITY;
      for(int w=0;w<V;++w) {
        float sum=0;for(int c=0;c<K;++c) sum=sum+bf(table[previous*K+c])*bf(weight[w*K+c]);
        float value=initial[i*V+w]+sum;
        worst=std::max(worst,std::abs(value-logits[i*V+w]));
        if(std::abs(value-logits[i*V+w])>1e-6f) return false;
        if(value>best_value) {best_value=value;best=w;}
      }
      if(!random && tokens[i+1]!=best) return false;
      if(tokens[i+1]<0 || tokens[i+1]>=V) return false;
      float score=0;for(int c=0;c<H;++c) score=score+bf(hidden[i*H+c])*bf(cw[c]);
      for(int c=0;c<K;++c) score=score+bf(table[previous*K+c])*bf(cw[H+c]);
      worst=std::max(worst,std::abs(score-confidence[i]));
      if(std::abs(score-confidence[i])>1e-6f) return false;
      previous=tokens[i+1];
    }
    return true;
  };
  for(int n : {0,1,3,5,6}) for(int seed=-1;seed<=V;++seed) {
    initialize(n,seed,0);submit(false);if(!compare(n,seed,false)) return 3;++cases;
  }
  for(int sequence=0;sequence<64;++sequence) {
    initialize(R,sequence%V,sequence);submit(true);if(!compare(R,sequence%V,true)) return 4;
    std::vector<int> saved(tokens,tokens+R+1);
    initialize(R,sequence%V,sequence);submit(true);
    if(!std::equal(saved.begin(),saved.end(),tokens)) return 5;
    initialize(R,sequence%V,sequence+1000);submit(true);
    if(!compare(R,sequence%V,true)) return 6;
    if(!std::equal(saved.begin(),saved.end(),tokens)) ++changed;
    cases+=3;
  }
  if(changed<32) return 7;
  // Ties choose the lowest vocabulary index, not a reduction-tree accident.
  for(int i=0;i<V*K;++i) weight[i]=__float2bfloat16_rn(0.f);
  std::fill(initial.begin(),initial.end(),0.f);
  initialize(R,3,0);submit(false);
  for(int i=1;i<=R;++i) if(tokens[i]!=0) return 8;
  ++cases;
  check(cudaFree(count));check(cudaFree(row));check(cudaFree(tokens));check(cudaFree(rng));
  check(cudaFree(hidden));check(cudaFree(table));check(cudaFree(weight));check(cudaFree(cw));
  check(cudaFree(embeds));check(cudaFree(logits));check(cudaFree(confidence));
  check(cudaFree(main_counts));check(cudaFree(start));check(cudaFree(seed));
  check(cudaFree(draft_counts));check(cudaFree(ids));check(cudaFree(positions));
  std::printf("sequential-markov passed cases=%d max_abs_error=%.9g changed_sequences=%d\n",cases,worst,changed);
  return 0;
}
