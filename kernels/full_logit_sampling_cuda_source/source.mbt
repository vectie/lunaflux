///|
/// AOT-only F32-logit sampler. Parameters and results use the existing 40-byte
/// sampling row / 8-byte token-status representation. No logits cross the host
/// boundary. A caller executes init, every IR matching, weights, then finish.
pub fn source(program : @physical.FullVocabularySamplingProgram) -> String {
  let (vocabulary, padded, rows) = program.geometry()
  let offset = program.weight_offset_bytes()
  let prefix =
    $|#include <stdint.h>
    $|#include <math.h>
    $|#include <limits.h>
    $|#define V \{vocabulary}
    $|#define N \{padded}
    $|#define R \{rows}
    $|struct P { int mode, top_k; double temperature, top_p; uint64_t seed; int sample_index, flags; };
    $|static_assert(sizeof(P)==40, "sampling ABI");
    $|struct C { float value; int token; };
    $|static_assert(sizeof(C)==8, "candidate ABI");
    $|__device__ bool before(C a,C b) { return a.value>b.value || (a.value==b.value && a.token<b.token); }
    $|__device__ const P* parameters(const int* control) { return (const P*)(control+2); }
    $|__device__ uint64_t counter(uint64_t seed,int index) {
    $|  uint64_t v=seed+((uint64_t)index+1ULL)*0x9e3779b97f4a7c15ULL;
    $|  v=(v^(v>>30))*0xbf58476d1ce4e5b9ULL; v=(v^(v>>27))*0x94d049bb133111ebULL; v^=v>>31;
    $|  return v==0ULL?0x9e3779b97f4a7c15ULL:v;
    $|}
    $|extern "C" __global__ void luna_full_sampling_init(const int* control,const float* logits,void* workspace,int* results) {
    $|  int r=blockIdx.x,t=threadIdx.x; if(r>=control[0] || r>=R) return;
    $|  P p=parameters(control)[r];
    $|  bool bad=p.mode<0 || p.mode>1 || (p.mode==1 && (!(p.temperature>0.0) || !isfinite(p.temperature) || p.sample_index<0 || p.seed==0ULL || (p.flags&~3) || ((p.flags&1) && p.top_k<=0) || ((p.flags&2) && (!(p.top_p>0.0) || p.top_p>1.0 || !isfinite(p.top_p)))));
    $|  if(bad) { if(t==0) {results[2*r]=-1;results[2*r+1]=-3;} return; }
    $|  C* c=(C*)workspace+(uint64_t)r*N; const float* x=logits+(uint64_t)r*V;
    $|  __shared__ C best[256]; __shared__ int invalid[256];
    $|  C b={-INFINITY,INT_MAX}; int e=INT_MAX;
    $|  for(int i=t;i<N;i+=256) { C v={-INFINITY,INT_MAX}; if(i<V) {v.value=x[i];v.token=i;if(!isfinite(v.value)) e=min(e,i);else if(before(v,b)) b=v;} c[i]=v; }
    $|  best[t]=b;invalid[t]=e;__syncthreads();
    $|  for(int s=128;s>0;s>>=1) {if(t<s) {if(before(best[t+s],best[t])) best[t]=best[t+s];invalid[t]=min(invalid[t],invalid[t+s]);}__syncthreads();}
    $|  if(t==0) {results[2*r]=invalid[0]==INT_MAX?best[0].token:-1;results[2*r+1]=invalid[0]==INT_MAX?-1:invalid[0];}
    $|}
    $|extern "C" __global__ void luna_full_sampling_sort(const int* control,void* workspace,const int* stage,const int* results) {
    $|  int r=blockIdx.y; if(r>=control[0] || r>=R || parameters(control)[r].mode==0 || results[2*r+1]!=-1) return;
    $|  int i=blockIdx.x*blockDim.x+threadIdx.x,j=i^stage[1]; if(i>=N || j<=i) return;
    $|  C* c=(C*)workspace+(uint64_t)r*N; C a=c[i],b=c[j];
    $|  if(before(a,b)!=((i&stage[0])==0)) {c[i]=b;c[j]=a;}
    $|}
    $|extern "C" __global__ void luna_full_sampling_weights(const int* control,void* workspace,const int* results) {
    $|  int r=blockIdx.y; if(r>=control[0] || r>=R || parameters(control)[r].mode==0 || results[2*r+1]!=-1) return;
    $|  P p=parameters(control)[r]; int k=(p.flags&1)?min(p.top_k,V):V; int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=k)return;
    $|  C* c=(C*)workspace+(uint64_t)r*N; double* w=(double*)((char*)workspace+\{offset}ULL)+(uint64_t)r*V;
    $|  w[i]=exp(((double)c[i].value-(double)c[0].value)/p.temperature);
    $|}
    $|extern "C" __global__ void luna_full_sampling_finish(const int* control,void* workspace,int* results) {
    $|  int r=blockIdx.x;if(r>=control[0] || r>=R || threadIdx.x!=0 || parameters(control)[r].mode==0 || results[2*r+1]!=-1)return;
    $|  P p=parameters(control)[r]; int k=(p.flags&1)?min(p.top_k,V):V;
    $|  C* c=(C*)workspace+(uint64_t)r*N;double* w=(double*)((char*)workspace+\{offset}ULL)+(uint64_t)r*V;
  prefix + "\n" + finish_source(program) + "\n}\n"
}
