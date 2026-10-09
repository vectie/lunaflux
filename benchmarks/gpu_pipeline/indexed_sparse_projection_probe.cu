#include "indexed_sparse_projection.cuh"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cmath>

static void ck(cudaError_t e) {
  if(e!=cudaSuccess) { std::fprintf(stderr,"%s\n",cudaGetErrorString(e)); std::exit(2); }
}
static void require(bool b,int line) {
  if(!b) { std::fprintf(stderr,"assertion line %d\n",line); std::exit(3); }
}
#define CHECK(x) require((x),__LINE__)
using BF = __nv_bfloat16;
static BF bf(float x) { return __float2bfloat16_rn(x); }
static float f(BF x) { return __bfloat162float(x); }
template<class T> static T *alloc(int n) {
  T *p; ck(cudaMalloc(&p,n*sizeof(T))); ck(cudaMemset(p,0,n*sizeof(T))); return p;
}
template<class T> static void put(T *p,const std::vector<T>& v) {
  ck(cudaMemcpy(p,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice));
}
template<class T> static std::vector<T> get(T *p,int n) {
  std::vector<T> v(n); ck(cudaMemcpy(v.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost)); return v;
}
static void dense(const std::vector<BF>& x,const std::vector<BF>& w,
    std::vector<BF>& y,int rows,int in,int out) {
  y.resize(rows*out);
  for(int r=0;r<rows;++r) for(int c=0;c<out;++c) {
    float sum=0; for(int j=0;j<in;++j) { volatile float p=f(x[r*in+j])*f(w[c*in+j]); sum=sum+p; }
    y[r*out+c]=bf(sum);
  }
}
static void rms(const std::vector<BF>& x,const std::vector<BF>& w,
    std::vector<BF>& y,int rows,int width) {
  y.resize(rows*width);
  for(int r=0;r<rows;++r) {
    float p[256]={}; for(int c=0;c<width;++c) p[c]=f(x[r*width+c])*f(x[r*width+c]);
    for(int stride=128;stride;stride/=2) for(int c=0;c<stride;++c) p[c]=p[c]+p[c+stride];
    const float inv=1.0f/std::sqrt(p[0]/width+0.00001f);
    for(int c=0;c<width;++c) y[r*width+c]=bf((f(x[r*width+c])*inv)*f(w[c]));
  }
}

int main() {
  ck(cudaSetDevice(0));
  const int widths[13]={4,4,8,2,2,16,8,8,8,4,4,2,4};
  const int weights[12]={32,4,32,16,2,32,32,32,4,4,16,32};
  BF *b[13],*w[12],*hidden=alloc<BF>(64);
  int32_t *counts=alloc<int32_t>(5);
  std::vector<BF> hw[12],cpu[13],input(64);
  for(int i=0;i<13;++i) b[i]=alloc<BF>(8*widths[i]);
  for(int i=0;i<12;++i) {
    w[i]=alloc<BF>(weights[i]); hw[i].resize(weights[i]);
    for(int j=0;j<weights[i];++j) hw[i][j]=bf(((j*3+i)%9-4)*0.125f);
    put(w[i],hw[i]);
  }
  for(int j=0;j<64;++j) input[j]=bf(((j*5)%17-8)*0.0625f);
  put(hidden,input);
  const auto launch=[&]() {
    sparse_projection_q_a<<<1,256>>>(counts,hidden,w[0],b[0]);
    sparse_projection_q_norm<<<8,256>>>(counts,b[0],w[1],b[1]);
    sparse_projection_q_b<<<1,256>>>(counts,b[1],w[2],b[2]);
    sparse_projection_kv_a<<<1,256>>>(counts,hidden,w[3],b[3]);
    sparse_projection_kv_norm<<<8,256>>>(counts,b[3],w[4],b[4]);
    sparse_projection_kv_b<<<1,256>>>(counts,b[4],w[5],b[5]);
    sparse_projection_kv_split<<<1,256>>>(counts,b[5],b[6],b[7]);
    sparse_projection_index_q<<<1,256>>>(counts,b[1],w[6],b[8]);
    sparse_projection_index_k<<<1,256>>>(counts,hidden,w[7],b[9]);
    sparse_projection_index_norm<<<8,1>>>(counts,b[9],w[8],w[9],b[10]);
    sparse_projection_index_weights<<<1,256>>>(counts,hidden,w[10],b[11]);
    sparse_projection_pool_gate<<<1,256>>>(counts,hidden,w[11],b[12]);
  };
  const auto step=[&](int rows) {
    put(counts,std::vector<int32_t>{0,0,1,rows,0});
    launch(); ck(cudaGetLastError()); ck(cudaDeviceSynchronize());
  };
  step(8);
  dense(input,hw[0],cpu[0],8,8,4); rms(cpu[0],hw[1],cpu[1],8,4);
  dense(cpu[1],hw[2],cpu[2],8,4,8);
  dense(input,hw[3],cpu[3],8,8,2); rms(cpu[3],hw[4],cpu[4],8,2);
  dense(cpu[4],hw[5],cpu[5],8,2,16);
  cpu[6].resize(64); cpu[7].resize(64);
  for(int r=0;r<8;++r) for(int h=0;h<2;++h) for(int c=0;c<4;++c) {
    cpu[6][r*8+h*4+c]=cpu[5][r*16+h*8+c];
    cpu[7][r*8+h*4+c]=cpu[5][r*16+h*8+4+c];
  }
  dense(cpu[1],hw[6],cpu[8],8,4,8); dense(input,hw[7],cpu[9],8,8,4);
  cpu[10].resize(32);
  for(int r=0;r<8;++r) {
    float mean=0; for(int c=0;c<4;++c) mean+=f(cpu[9][r*4+c]); mean/=4;
    float variance=0; for(int c=0;c<4;++c) { const float d=f(cpu[9][r*4+c])-mean; volatile float p=d*d; variance+=p; }
    const float inv=1.0f/std::sqrt(variance/4+0.00001f);
    for(int c=0;c<4;++c) cpu[10][r*4+c]=bf(((f(cpu[9][r*4+c])-mean)*inv)*f(hw[8][c])+f(hw[9][c]));
  }
  dense(input,hw[10],cpu[11],8,8,2); dense(input,hw[11],cpu[12],8,8,4);
  float maxabs=0;
  std::vector<BF> saved[13];
  for(int i=0;i<13;++i) {
    saved[i]=get(b[i],8*widths[i]);
    for(int j=0;j<8*widths[i];++j) maxabs=std::fmax(maxabs,std::fabs(f(saved[i][j])-f(cpu[i][j])));
  }
  CHECK(maxabs==0.0f);
  step(1);
  for(int i=0;i<13;++i) {
    const auto live=get(b[i],widths[i]);
    CHECK(std::memcmp(live.data(),saved[i].data(),widths[i]*sizeof(BF))==0);
  }
  step(0); // no projection writes on idle
  for(int i=0;i<13;++i) { auto live=get(b[i],widths[i]); CHECK(std::memcmp(live.data(),saved[i].data(),widths[i]*2)==0); }
  int32_t *reset=alloc<int32_t>(1), *history=alloc<int32_t>(5), *valid=alloc<int32_t>(32);
  int32_t *append=alloc<int32_t>(3), *positions=alloc<int32_t>(8);
  int32_t *raw=alloc<int32_t>(32), *pv=alloc<int32_t>(8), *pc=alloc<int32_t>(1), *selected=alloc<int32_t>(88);
  BF *hi=alloc<BF>(128), *hg=alloc<BF>(128), *hk=alloc<BF>(256), *hv=alloc<BF>(256);
  BF *ape=alloc<BF>(16), *pooled=alloc<BF>(32), *out=alloc<BF>(64);
  const auto retained=[&](int rows,bool clear) {
    put(counts,std::vector<int32_t>{0,0,1,rows,0}); put(reset,std::vector<int32_t>{clear?1:0});
    projected_history_begin<<<1,1>>>(counts,reset,history,valid,append,positions);
    launch();
    projected_history_copy<<<1,256>>>(append,b[10],b[12],b[6],b[7],hi,hg,hk,hv);
    projected_history_publish<<<1,1>>>(append,history,valid);
    projected_attention_pool<<<1,1>>>(history,hi,hg,valid,ape,pooled,raw,pv,pc);
    projected_attention_index<<<1,1>>>(counts,positions,b[8],b[11],pooled,raw,pv,pc,valid,selected);
    projected_attention_attention<<<1,1>>>(counts,positions,history+3,valid,b[2],hk,hv,selected,out);
    ck(cudaGetLastError()); ck(cudaDeviceSynchronize());
    CHECK(get(append,3)[2]==0);
    const auto pos=get(positions,8), indices=get(selected,88);
    const auto key=get(hk,256), value=get(hv,256), query=get(b[2],64), output=get(out,64);
    for(int r=0;r<rows;++r) {
      std::vector<int> ids;
      for(int j=0;j<11;++j) if(indices[r*11+j]>=0) {
        CHECK(indices[r*11+j]<=pos[r]);
        for(auto previous:ids) CHECK(previous!=indices[r*11+j]);
        ids.push_back(indices[r*11+j]);
      }
      CHECK((int)ids.size()==pos[r]+1); // all visible keys fit this fixture's top-k/tail
      for(int h=0;h<2;++h) {
        std::vector<float> scores; float maximum=-INFINITY;
        for(auto id:ids) {
          float sum=0; for(int c=0;c<4;++c) { volatile float p=f(query[r*8+h*4+c])*f(key[id*8+h*4+c]); sum+=p; }
          scores.push_back(sum*0.5f); maximum=std::fmax(maximum,scores.back());
        }
        float denom=0; for(auto score:scores) denom+=std::exp(score-maximum);
        for(int c=0;c<4;++c) {
          float sum=0;
          for(int j=0;j<(int)ids.size();++j) { volatile float p=f(bf(std::exp(scores[j]-maximum)/denom))*f(value[ids[j]*8+h*4+c]); sum+=p; }
          const auto expected=bf(sum);
          CHECK(std::memcmp(&output[r*8+h*4+c], &expected, 2)==0);
        }
      }
    }
  };
  retained(8,true); CHECK(get(history,5)[3]==8);
  auto retained_keys=get(hk,64), retained_values=get(hv,64), retained_index=get(hi,32);
  CHECK(std::memcmp(retained_keys.data(),saved[6].data(),128)==0);
  CHECK(std::memcmp(retained_values.data(),saved[7].data(),128)==0);
  CHECK(std::memcmp(retained_index.data(),saved[10].data(),64)==0);
  retained(1,false); CHECK(get(history,5)[3]==9);
  retained(0,false); CHECK(get(history,5)[3]==9);
  for(auto v:get(out,64)) CHECK(f(v)==0.0f);
  for(void *p:std::vector<void*>{reset,history,valid,append,positions,raw,pv,pc,selected,hi,hg,hk,hv,ape,pooled,out}) ck(cudaFree(p));
  for(auto p:b) ck(cudaFree(p)); for(auto p:w) ck(cudaFree(p));
  ck(cudaFree(hidden)); ck(cudaFree(counts)); ck(cudaDeviceSynchronize());
  std::puts("maxabs=0 projection_prefill_decode_bitwise=true affine_index_norm=true hidden_to_retained_attention=true all_allocations_released=true");
}
