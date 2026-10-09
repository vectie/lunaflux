#include "indexed_sparse.cuh"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static void ck(cudaError_t e) {
  if(e!=cudaSuccess) { std::fprintf(stderr,"%s\n",cudaGetErrorString(e)); std::exit(2); }
}
static void require(bool ok, int line) {
  if(!ok) { std::fprintf(stderr,"assertion line %d\n",line); std::exit(3); }
}
#define CHECK(x) require((x),__LINE__)
template<class T> static T *allocate(int n) {
  T *p; ck(cudaMalloc(&p,n*sizeof(T))); ck(cudaMemset(p,0,n*sizeof(T))); return p;
}
template<class T> static void put(T *p, const std::vector<T> &v) {
  ck(cudaMemcpy(p,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice));
}
template<class T> static std::vector<T> get(T *p,int n) {
  std::vector<T> v(n); ck(cudaMemcpy(v.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost)); return v;
}
static __nv_bfloat16 bf(float x) { return __float2bfloat16_rn(x); }
static bool same(__nv_bfloat16 a,__nv_bfloat16 b) { return std::memcmp(&a,&b,2)==0; }

// Exactly the owned history and borrowed projection/frame regions of the
// MoonBit binder. Two instances emulate independent request owners.
struct Request {
  int32_t *counts=allocate<int32_t>(5), *reset=allocate<int32_t>(1);
  int32_t *history=allocate<int32_t>(5), *valid=allocate<int32_t>(32);
  int32_t *append=allocate<int32_t>(3), *positions=allocate<int32_t>(8);
  __nv_bfloat16 *ik=allocate<__nv_bfloat16>(32), *gate=allocate<__nv_bfloat16>(32);
  __nv_bfloat16 *keys=allocate<__nv_bfloat16>(32), *values=allocate<__nv_bfloat16>(32);
  __nv_bfloat16 *hi=allocate<__nv_bfloat16>(128), *hg=allocate<__nv_bfloat16>(128);
  __nv_bfloat16 *hk=allocate<__nv_bfloat16>(128), *hv=allocate<__nv_bfloat16>(128);
  __nv_bfloat16 *iq=allocate<__nv_bfloat16>(32), *iw=allocate<__nv_bfloat16>(8);
  __nv_bfloat16 *q=allocate<__nv_bfloat16>(64), *ape=allocate<__nv_bfloat16>(16);
  __nv_bfloat16 *pool=allocate<__nv_bfloat16>(32), *out=allocate<__nv_bfloat16>(64);
  int32_t *raw=allocate<int32_t>(32), *pv=allocate<int32_t>(8);
  int32_t *pc=allocate<int32_t>(1), *selected=allocate<int32_t>(88);
  std::vector<__nv_bfloat16> expected_i,expected_g,expected_v;

  void step(int live,bool clear,int tag) {
    int base=clear?0:(int)expected_v.size()/4;
    if(clear) { expected_i.clear(); expected_g.clear(); expected_v.clear(); }
    put(counts,std::vector<int32_t>{0,0,1,live,0}); put(reset,std::vector<int32_t>{clear?1:0});
    std::vector<__nv_bfloat16> a(32,bf(0)),b(32,bf(0)),v(32,bf(0));
    for(int row=0;row<live;++row) for(int c=0;c<4;++c) {
      a[row*4+c]=bf((base+row+1)*(c+1)*0.03125f);
      b[row*4+c]=bf((base+row+1)*0.0625f);
      v[row*4+c]=bf((tag+base+row+1)*(c+1)*0.125f);
    }
    put(ik,a); put(gate,b); put(values,v);
    sparse_prefill_history_begin<<<1,1>>>(counts,reset,history,valid,append,positions);
    sparse_prefill_history_copy<<<1,256>>>(append,ik,gate,keys,values,hi,hg,hk,hv);
    sparse_prefill_history_publish<<<1,1>>>(append,history,valid);
    sparse_prefill_pool<<<1,1>>>(history,hi,hg,valid,ape,pool,raw,pv,pc);
    sparse_prefill_index<<<1,1>>>(counts,positions,iq,iw,pool,raw,pv,pc,valid,selected);
    sparse_prefill_attention<<<1,1>>>(counts,positions,history+3,valid,q,hk,hv,selected,out);
    ck(cudaGetLastError()); ck(cudaDeviceSynchronize());
    auto descriptor=get(append,3);
    if(base+live>32) {
      CHECK(descriptor[2]==1); CHECK(get(history,5)[3]==0);
      for(auto x:get(out,64)) CHECK(same(x,bf(0)));
      return;
    }
    CHECK(descriptor[0]==base && descriptor[1]==live && descriptor[2]==0);
    expected_i.insert(expected_i.end(),a.begin(),a.begin()+live*4);
    expected_g.insert(expected_g.end(),b.begin(),b.begin()+live*4);
    expected_v.insert(expected_v.end(),v.begin(),v.begin()+live*4);
    CHECK(get(history,5)[3]==base+live);
    auto mask=get(valid,32),pos=get(positions,8),indices=get(selected,88);
    auto output=get(out,64);
    for(int t=0;t<32;++t) CHECK(mask[t]==(t<base+live?1:0));
    auto ei=get(hi,128),eg=get(hg,128),ev=get(hv,128),ek=get(hk,128);
    for(int i=0;i<(base+live)*4;++i) {
      CHECK(same(ei[i],expected_i[i])); CHECK(same(eg[i],expected_g[i]));
      CHECK(same(ev[i],expected_v[i])); CHECK(same(ek[i],bf(0)));
    }
    for(int row=0;row<8;++row) {
      CHECK(pos[row]==(row<live?base+row:-1));
      if(row>=live) { for(int c=0;c<8;++c) CHECK(same(output[row*8+c],bf(0))); continue; }
      const int position=base+row, pools=(position+1)/4<2?(position+1)/4:2;
      const int tail=(position+1)%4, start=position+1-tail;
      const int available=(base+live)/4<2?(base+live)/4:2;
      std::vector<int32_t> selected(11,-1);
      for(int i=0;i<pools*4;++i) selected[i]=i;
      for(int i=0;i<tail;++i) selected[available*4+i]=start+i;
      for(int i=0;i<11;++i) CHECK(indices[row*11+i]==selected[i]);
      const float prob=__bfloat162float(bf(1.0f/(pools*4+tail)));
      for(int head=0;head<2;++head) for(int c=0;c<4;++c) {
        float sum=0;
        for(int i=0;i<11;++i) if(indices[row*11+i]>=0) {
          volatile float product=prob*__bfloat162float(expected_v[indices[row*11+i]*4+c]);
          sum=sum+product;
        }
        CHECK(same(output[(row*2+head)*4+c],bf(sum)));
      }
    }
  }

  void close() {
    for(void *p:std::vector<void*>{counts,reset,history,valid,append,positions,ik,gate,keys,values,
        hi,hg,hk,hv,iq,iw,q,ape,pool,out,raw,pv,pc,selected}) ck(cudaFree(p));
  }
};

int main() {
  ck(cudaSetDevice(0));
  Request a,b,c;
  a.step(8,true,0); b.step(4,true,100);
  const auto before=get(b.hv,128);
  a.step(4,false,0); a.step(1,false,0);
  const auto decoded=get(a.out,8);
  c.step(8,true,0); c.step(5,false,0);
  const auto prefilled=get(c.out,64);
  CHECK(std::memcmp(decoded.data(),prefilled.data()+4*8,16)==0);
  auto after=get(b.hv,128); CHECK(std::memcmp(before.data(),after.data(),256)==0);
  b.step(1,false,100);
  a.step(0,false,0); // idle preserves history while clearing query output
  a.step(8,false,0); a.step(8,false,0); a.step(3,false,0); // exactly full
  a.step(1,false,0); // overflow reports a device error, never truncates silently
  a.step(2,true,300); // slot reuse erases validity of all previous tokens
  a.step(1,false,300); b.step(0,false,100);
  a.close(); b.close(); c.close(); ck(cudaDeviceSynchronize());
  std::puts("maxabs=0 retained_prefill_decode_bitwise=true independent_requests=true reset_reuse=true idle_preserved=true capacity_error=true all_allocations_released=true");
}
