// Whole-output coverage oracle for the diagnostic attention substitution.
// Zero Q/K gives a known causal mean; shuffled pages, unequal row lengths,
// nonconstant values, poisoned outputs/workspace, and untouched KV are checked.
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cstring>
#define CHECK(x) do {auto code=(x);if(int(code)){std::fprintf(stderr,"%s: %d\n",#x,int(code));std::exit(1);}}while(0)
#define LF_REFERENCE_STANDALONE 1
#include "reference_attention_swap.cuh"
template<class T> struct Buffer {
  T* p;size_t n;
  explicit Buffer(size_t count):n(count){CHECK(cudaMalloc(&p,n*sizeof(T)));}
  ~Buffer(){CHECK(cudaFree(p));}
  void put(const std::vector<T>& h){if(h.size()!=n) std::abort();CHECK(cudaMemcpy(p,h.data(),n*sizeof(T),cudaMemcpyHostToDevice));}
  std::vector<T> get(){std::vector<T> h(n);CHECK(cudaMemcpy(h.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost));return h;}
};
static float value(int row,int head,int pos,int dim) {
  return row*0.03125f+head*0.015625f+(pos%16)*0.0078125f+(dim%4)*0.001953125f;
}
static void test(const std::vector<int>& queries,int prefill) {
  int b=queries.size(),tokens=0,pages=0;
  std::vector<int> offsets{0},po{0},lengths,positions,ids;
  for(int row=0;row<b;row++) {
    int length=8192+row*3;
    lengths.push_back(length);tokens+=queries[row];offsets.push_back(tokens);
    pages+=(length+7)/8;po.push_back(pages);
    for(int q=0;q<queries[row];q++) positions.push_back(length-queries[row]+q);
  }
  for(int i=0;i<pages;i++) ids.push_back(pages-1-i);
  std::vector<int> counts{prefill,b-prefill,b,tokens,pages};
  Buffer<int> dc(5),dpos(tokens),doff(b+1),dlen(b),dpo(b+1),did(pages);
  dc.put(counts);dpos.put(positions);doff.put(offsets);dlen.put(lengths);dpo.put(po);did.put(ids);
  Buffer<__nv_bfloat16> q(size_t(tokens)*4096),k(size_t(pages)*8192),v(size_t(pages)*8192),out(size_t(tokens+7)*2048);
  std::vector<__nv_bfloat16> qh(q.n,__float2bfloat16(0)),kh(k.n,__float2bfloat16(0)),vh(v.n,__float2bfloat16(-99));
  for(int row=0;row<b;row++) for(int pos=0;pos<lengths[row];pos++) for(int head=0;head<8;head++) for(int d=0;d<128;d++) {
    size_t index=size_t(ids[po[row]+pos/8])*8192+(pos%8)*1024+head*128+d;
    vh[index]=__float2bfloat16(value(row,head,pos,d));
  }
  q.put(qh);k.put(kh);v.put(vh);Buffer<unsigned char> workspace(256*1024);
  AttentionArgs args{dc.p,dpos.p,doff.p,dlen.p,dpo.p,did.p,q.p,k.p,v.p};
  size_t checked=0;
  for(float poison:{1024.f,-1024.f}) {
    std::vector<__nv_bfloat16> oh(out.n,__float2bfloat16(poison));out.put(oh);
    CHECK(cudaMemset(workspace.p,0xff,workspace.n));
    reference_attention(args,out.p,workspace.p,nullptr);CHECK(cudaDeviceSynchronize());
    auto actual=out.get();
    for(int row=0;row<b;row++) for(int token=offsets[row];token<offsets[row+1];token++) {
      int history=positions[token]+1;
      double meanpos=(history/16)*120;
      for(int p=0;p<history%16;p++) meanpos+=p;
      meanpos/=history;
      for(int head=0;head<16;head++) for(int d=0;d<128;d++) {
        double expected=row*.03125+(head/2)*.015625+meanpos*.0078125+(d%4)*.001953125;
        float got=__bfloat162float(actual[size_t(token)*2048+head*128+d]);
        if(!std::isfinite(got)||std::abs(got-expected)>.004) {
          std::fprintf(stderr,"mismatch prefill=%d row=%d token=%d head=%d d=%d got=%g want=%g\n",prefill,row,token,head,d,got,expected);std::exit(2);
        } ++checked;
      }
    }
    for(size_t i=size_t(tokens)*2048;i<actual.size();i++) if(__bfloat162float(actual[i])!=poison) std::abort();
  }
  auto afterk=k.get(),afterv=v.get();
  if(std::memcmp(afterk.data(),kh.data(),kh.size()*2)||std::memcmp(afterv.data(),vh.data(),vh.size()*2)) std::abort();
  std::printf("prefill=%d rows=%d tokens=%d checked=%zu padding_unchanged=1 kv_unchanged=1 passed=1\n",prefill,b,tokens,checked);
}
int main(int argc,char**) {
  CHECK(cudaSetDevice(0));configure_reference_attention();
  if(argc>1) {test({91,1,1,1,1,1,1,1},1);test({1,1,1,1,1,1,1,1},0);}
  else {test({2048},1);test({2030,1,1,1,1,1,1,1},1);test({91,1,1,1,1,1,1,1},1);test({1,1,1,1,1,1,1,1},0);test({1},0);}
  CHECK(cudaDeviceSynchronize());CHECK(cudaDeviceReset());
}
