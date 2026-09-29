// Source-fixture oracle for grouped address ownership and double-slot effects.
// Include one unchanged repair-export decode source using -include.
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

static void ck(cudaError_t status) {
  if (status != cudaSuccess) { std::fprintf(stderr, "%s\n", cudaGetErrorString(status)); std::exit(2); }
}
template<class T> struct Buffer {
  T* p;
  size_t count;
  explicit Buffer(const std::vector<T>& host): count(host.size()) {
    ck(cudaMalloc(&p, count * sizeof(T)));
    ck(cudaMemcpy(p, host.data(), count * sizeof(T), cudaMemcpyHostToDevice));
  }
  ~Buffer() { ck(cudaFree(p)); }
  std::vector<T> read() {
    std::vector<T> host(count);
    ck(cudaMemcpy(host.data(), p, count * sizeof(T), cudaMemcpyDeviceToHost));
    return host;
  }
};
static float fp(__nv_bfloat16 x) { return __bfloat162float(x); }
static void run(int context, int batch, bool mixed, bool invalid=false) {
  constexpr int heads=16, kv_heads=4, dim=128, width=3072, stride=8192, capacity=256;
  const int prefix=mixed?3:0, prefill=mixed?1:0, rows=batch+prefill, tokens=batch+prefix;
  const int pages=(context+15)/16;
  // Metadata is bounded by the emitted ABI, independently of cache storage.
  if (batch*pages+prefill>256) return;
  std::vector<int> positions(tokens,context-1), offsets(rows+1), lengths(rows,context), tables(rows+1), ids;
  if (mixed) { offsets[1]=prefix; lengths[0]=prefix; tables[1]=1; ids.push_back(0); }
  for(int b=0;b<batch;++b) {
    offsets[prefill+b]=prefix+b; offsets[prefill+b+1]=prefix+b+1;
    tables[prefill+b]=int(ids.size());
    for(int p=0;p<pages;++p) ids.push_back((p*73+b*19)%capacity);
    tables[prefill+b+1]=int(ids.size());
  }
  const std::vector<int> counts={prefill,batch,rows,tokens,int(ids.size())};
  std::vector<__nv_bfloat16> q(tokens*width), k(capacity*stride), v(k.size()), out(tokens*heads*dim,__float2bfloat16_rn(-99));
  for(size_t i=0;i<q.size();++i) q[i]=__float2bfloat16_rn(float(int(i*17%127)-63)/64);
  for(size_t i=0;i<k.size();++i) { k[i]=__float2bfloat16_rn(float(int(i*13%113)-56)/64); v[i]=__float2bfloat16_rn(float(int(i*19%109)-54)/64); }
  std::vector<double> expected(batch*heads*dim);
  if(!invalid) for(int b=0;b<batch;++b) for(int h=0;h<heads;++h) {
    std::vector<double> scores(context); double maximum=-INFINITY, denominator=0;
    for(int p=0;p<context;++p) {
      size_t base=size_t(ids[tables[prefill+b]+p/16])*stride+(p%16)*kv_heads*dim+(h/4)*dim;
      double dot=0; for(int c=0;c<dim;++c) dot+=double(fp(q[(prefix+b)*width+h*dim+c]))*fp(k[base+c]);
      scores[p]=dot/std::sqrt(double(dim)); maximum=std::max(maximum,scores[p]);
    }
    for(double& x:scores) { x=std::exp(x-maximum); denominator+=x; }
    for(int c=0;c<dim;++c) {
      double sum=0;
      for(int p=0;p<context;++p) {
        size_t base=size_t(ids[tables[prefill+b]+p/16])*stride+(p%16)*kv_heads*dim+(h/4)*dim;
        sum+=scores[p]*fp(v[base+c]);
      }
      expected[(b*heads+h)*dim+c]=sum/denominator;
    }
  }
  if(invalid) ids[tables[prefill]]=capacity;
  Buffer<int> dc(counts), dp(positions), dr(offsets), dl(lengths), dt(tables), di(ids);
  Buffer<__nv_bfloat16> dq(q), dk(k), dv(v), dout(out);
  ck(cudaFuncSetAttribute(owned_decode,cudaFuncAttributeMaxDynamicSharedMemorySize,LF_SHARED_MEMORY_BYTES));
  owned_decode<<<dim3(32,kv_heads),LF_PROBE_BLOCK_THREADS,LF_SHARED_MEMORY_BYTES>>>(dc.p,dp.p,dr.p,dl.p,dt.p,di.p,dq.p,dout.p,dk.p,dv.p);
  ck(cudaGetLastError()); ck(cudaDeviceSynchronize());
  auto got=dout.read(); double error=0;
  if(invalid) {
    // Transfer validity rejects the entire CTA after draining its writers;
    // this existing ABI leaves output untouched for an invalid cache page.
    for(size_t i=0;i<got.size();++i) if(fp(got[i])!=-99) {
      std::fprintf(stderr,"invalid page modified output at %zu\n",i); std::exit(5);
    }
  } else for(size_t i=0;i<expected.size();++i) {
    float actual=fp(got[prefix*heads*dim+i]);
    if(!std::isfinite(actual)) std::exit(3);
    error=std::max(error,std::fabs(actual-expected[i]));
  }
  if(error>0.004) { std::fprintf(stderr,"oracle error=%g\n",error); std::exit(3); }
  for(int i=0;i<prefix*heads*dim;++i) if(fp(got[i])!=-99) std::exit(4);
  auto after_k=dk.read(), after_v=dv.read();
  for(size_t i=0;i<k.size();++i) if(fp(after_k[i])!=fp(k[i]) || fp(after_v[i])!=fp(v[i])) std::exit(4);
  std::printf("context=%d batch=%d mixed=%d invalid=%d max_abs_error=%g passed\n",context,batch,int(mixed),int(invalid),error);
}
int main(int argc,char**) {
  for(int context:{1,31,32,33,63,64,65,129,257}) run(context,1,false);
  run(65,2,true); run(257,8,false); run(65,1,false,true);
  if(argc==1) run(4096,1,false);
  std::puts("decode_oracle=passed kv_unchanged=true");
}
