// Offline actual-source test, independent mathematical oracle; no serving dependency.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "latent.cuh"
static void ck(cudaError_t e){if(e!=cudaSuccess){fprintf(stderr,"%s\n",cudaGetErrorString(e));exit(2);}}
template<class T> struct Buffer {
 T *p;size_t n;explicit Buffer(size_t n):n(n){ck(cudaMalloc(&p,n*sizeof(T)));ck(cudaMemset(p,0,n*sizeof(T)));}
 ~Buffer(){ck(cudaFree(p));}
 void put(const std::vector<T>& v){if(v.size()!=n)exit(3);ck(cudaMemcpy(p,v.data(),n*sizeof(T),cudaMemcpyHostToDevice));}
 std::vector<T> get(){std::vector<T> v(n);ck(cudaMemcpy(v.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost));return v;}
};
static float f(__nv_bfloat16 x){return __bfloat162float(x);}
static __nv_bfloat16 bf(double x){return __float2bfloat16_rn(float(x));}
static std::vector<__nv_bfloat16> fixture(size_t n,int seed){std::vector<__nv_bfloat16> v(n);for(size_t i=0;i<n;++i)v[i]=bf((int((i*37+seed)%127)-63)/256.);return v;}
static void bound(double actual,double expected,double absolute,int n,const char *stage,int i){
 double gamma=n*std::ldexp(1.,-24)/(1-n*std::ldexp(1.,-24));
 double allowed=absolute*gamma+fabs(expected)/128.+1.e-7;
 if(!std::isfinite(actual)||fabs(actual-expected)>allowed){fprintf(stderr,"%s[%d] %.12g vs %.12g bound %.12g\n",stage,i,actual,expected,allowed);exit(4);}
}
int main(){
 {
 constexpr int rows=17,heads=4,kh=2,d=32,rank=64,history=64,selected_width=35;
 auto q=fixture(rows*heads*d,7),w=fixture(kh*2*d*rank,13),cache=fixture(history*rank,19);
 Buffer<int> counts(5),positions(rows),key_count(1),valid(history),selected(rows*selected_width);
 Buffer<__nv_bfloat16> query(q.size()),weight(w.size()),latent(cache.size()),absorbed(rows*heads*rank),context(rows*heads*rank),out(rows*heads*d);
 Buffer<unsigned char> scratch(attention_scratch);
 query.put(q);weight.put(w);latent.put(cache);key_count.put({history});valid.put(std::vector<int>(history,1));
 std::vector<int> pos(rows);for(int i=0;i<rows;++i)pos[i]=history-rows+i;positions.put(pos);
 for(int live:{0,1,15,16,17}){
  counts.put({0,0,1,live,0});std::vector<int> ids(rows*selected_width,-1);
  for(int row=0;row<live;++row)for(int i=0;i<32;++i)ids[row*selected_width+i]=i;
  selected.put(ids);
  void *a[]={&counts.p,&query.p,&weight.p,&absorbed.p};
  ck(cudaLaunchKernel((const void*)latent_projection_q_absorb,dim3(absorb_grid),dim3(128),a,0,nullptr));
  void *b[]={&counts.p,&positions.p,&key_count.p,&valid.p,&absorbed.p,&latent.p,&latent.p,&selected.p,&context.p,&scratch.p};
  ck(cudaLaunchKernel((const void*)latent_attention_attention_retained_scores_v1,dim3(attention_grid),dim3(attention_workers),b,0,nullptr));
  void *c[]={&counts.p,&context.p,&weight.p,&out.p};
  ck(cudaLaunchKernel((const void*)latent_value_expand,dim3(expand_grid),dim3(128),c,0,nullptr));ck(cudaDeviceSynchronize());
  auto aq=absorbed.get(),ctx=context.get(),y=out.get();
  for(int row=0;row<rows;++row)for(int h=0;h<heads;++h){
   for(int col=0;col<rank;++col){double dot=0,mass=0;for(int k=0;k<d&&row<live;++k){double p=double(f(q[(row*heads+h)*d+k]))*f(w[(h/(heads/kh)*2*d+k)*rank+col]);dot+=p;mass+=fabs(p);}bound(f(aq[(row*heads+h)*rank+col]),dot,mass,64,"query",(row*heads+h)*rank+col);}
   std::vector<double> logits(32);double mx=-INFINITY,den=0;
   for(int key=0;key<32;++key){double dot=0;for(int k=0;k<rank;++k)dot+=double(f(aq[(row*heads+h)*rank+k]))*f(cache[key*rank+k]);logits[key]=dot/std::sqrt(double(d));mx=std::max(mx,logits[key]);}
   for(double s:logits)den+=exp(s-mx);
   for(int col=0;col<rank;++col){double sum=0,mass=0;for(int key=0;key<32&&row<live;++key){double p=f(bf(exp(logits[key]-mx)/den))*double(f(cache[key*rank+col]));sum+=p;mass+=fabs(p);}bound(f(ctx[(row*heads+h)*rank+col]),sum,mass,512,"context",(row*heads+h)*rank+col);}
   for(int col=0;col<d;++col){double dot=0,mass=0;for(int k=0;k<rank&&row<live;++k){double p=double(f(ctx[(row*heads+h)*rank+k]))*f(w[(h/(heads/kh)*2*d+d+col)*rank+k]);dot+=p;mass+=fabs(p);}bound(f(y[(row*heads+h)*d+col]),dot,mass,128,"value",(row*heads+h)*d+col);}
  }
  // Determinism on identical inputs and state, without payload digests.
  ck(cudaLaunchKernel((const void*)latent_projection_q_absorb,dim3(absorb_grid),dim3(128),a,0,nullptr));
  ck(cudaLaunchKernel((const void*)latent_attention_attention_retained_scores_v1,dim3(attention_grid),dim3(attention_workers),b,0,nullptr));
  ck(cudaLaunchKernel((const void*)latent_value_expand,dim3(expand_grid),dim3(128),c,0,nullptr));ck(cudaDeviceSynchronize());auto repeated=out.get();
  for(size_t i=0;i<y.size();++i)if(f(y[i])!=f(repeated[i]))return 5;
  printf("latent actual-source live=%d query/attention/value independent_f64_bound=pass deterministic=true\n",live);
 }
 }
 ck(cudaDeviceReset());puts("closed=true");
}
