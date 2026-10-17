#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include "probe.cuh"
#define CK(x) do{auto e=(x);if(e!=cudaSuccess){fprintf(stderr,"%s: %s\n",#x,cudaGetErrorString(e));exit(2);}}while(0)
using Head=void(*)(bool,int,int*,__nv_bfloat16*,__nv_bfloat16*,float*);
using Greedy=void(*)(int,int*,int*,int*,float*,float*,int*);
using Join=void(*)(int*,float*,int*);
template<class T>T* alloc(size_t n){T*p;CK(cudaMalloc(&p,n*sizeof(T)));return p;}
template<class T>void put(T*p,const std::vector<T>&v){CK(cudaMemcpy(p,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice));}
template<class T>std::vector<T> get(T*p,size_t n){std::vector<T>v(n);CK(cudaMemcpy(v.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost));return v;}
template<class T>void equal(const std::vector<T>&a,const std::vector<T>&b){if(a.size()!=b.size()||memcmp(a.data(),b.data(),a.size()*sizeof(T))){fprintf(stderr,"bitwise mismatch\n");exit(3);}}
void run(int id,int v,int h,int o,Head head,Greedy greedy,Join join,bool timed){
 int split=v/2;auto counts=alloc<int>(5);auto count=alloc<int>(1);auto selected=alloc<int>(o);auto tokens=alloc<int>(o);
 auto x=alloc<__nv_bfloat16>(o*h);auto w=alloc<__nv_bfloat16>(size_t(v)*h);auto full=alloc<float>(o*v);
 float *parts[2]={alloc<float>(o*split),alloc<float>(o*(v-split))};auto candidates=alloc<float>(o*3*2);
 put(counts,std::vector<int>{1,0,1,64,0});put(count,std::vector<int>{o});put(selected,std::vector<int>(o,63));
 std::vector<__nv_bfloat16> hx(o*h);for(size_t i=0;i<hx.size();++i)hx[i]=__float2bfloat16_rn((int(i*37%211)-105)*.0078125f);put(x,hx);
 // Bounded upload: no extra host copy of the 1.27 GB vocabulary table.
 for(size_t start=0;start<size_t(v)*h;){size_t n=std::min(size_t(1048576),size_t(v)*h-start);std::vector<__nv_bfloat16> hw(n);for(size_t i=0;i<n;++i)hw[i]=__float2bfloat16_rn((int((start+i)*17%101)-50)*.001953125f);put(w+start,hw);start+=n;}
 head(false,0,count,x,w,full);head(true,0,count,x,w,parts[0]);head(true,1,count,x,w,parts[1]);CK(cudaDeviceSynchronize());
 auto expected=get(full,o*v);auto a=get(parts[0],o*split);auto b=get(parts[1],o*(v-split));
 for(int row=0;row<o;++row){equal(std::vector<float>(expected.begin()+row*v,expected.begin()+row*v+split),std::vector<float>(a.begin()+row*split,a.begin()+(row+1)*split));equal(std::vector<float>(expected.begin()+row*v+split,expected.begin()+(row+1)*v),std::vector<float>(b.begin()+row*(v-split),b.begin()+(row+1)*(v-split)));}
 auto compare=[&](){greedy(-1,counts,count,selected,full,candidates,tokens);CK(cudaDeviceSynchronize());auto reference=get(tokens,o);greedy(0,counts,count,selected,parts[0],candidates,tokens);greedy(1,counts,count,selected,parts[1],candidates+o*3,tokens);join(count,candidates,tokens);CK(cudaDeviceSynchronize());equal(reference,get(tokens,o));};
 compare();
 // Global ties cross a rank boundary. Every invalid logit invalidates greedy.
 if(id==0)for(int test=0;test<5;++test){std::vector<float> values(o*v,-1.f);for(int row=0;row<o;++row){values[row*v+1]=3.f;values[row*v+split+1]=3.f;if(test==1)values[row*v+v-1]=NAN;if(test==2)values[row*v]=INFINITY;}
  std::vector<float> pa,pb;for(int row=0;row<o;++row){pa.insert(pa.end(),values.begin()+row*v,values.begin()+row*v+split);pb.insert(pb.end(),values.begin()+row*v+split,values.begin()+(row+1)*v);}put(full,values);put(parts[0],pa);put(parts[1],pb);put(count,std::vector<int>{test==3?1:o});put(selected,std::vector<int>(o,test==4?64:63));CK(cudaMemset(tokens,0x5a,o*4));compare();}
 if(timed&&id==1){cudaEvent_t s,z;CK(cudaEventCreate(&s));CK(cudaEventCreate(&z));for(int trial=0;trial<5;++trial)for(int rank=-1;rank<2;++rank){CK(cudaEventRecord(s));for(int k=0;k<10;++k)head(rank>=0,rank,count,x,w,rank<0?full:parts[rank]);CK(cudaEventRecord(z));CK(cudaEventSynchronize(z));float ms;CK(cudaEventElapsedTime(&ms,s,z));printf("trial=%d head_rank=%d ms=%.9f (single_device_leaf_not_tp_e2e)\n",trial,rank,ms/10);}CK(cudaEventDestroy(s));CK(cudaEventDestroy(z));}
 CK(cudaFree(counts));CK(cudaFree(count));CK(cudaFree(selected));CK(cudaFree(tokens));CK(cudaFree(x));CK(cudaFree(w));CK(cudaFree(full));CK(cudaFree(parts[0]));CK(cudaFree(parts[1]));CK(cudaFree(candidates));printf("case=%d logits=bitwise greedy=bitwise released=true\n",id);
}
int main(int argc,char**argv){bool small=argc>1&&strcmp(argv[1],"small")==0;run(0,17,64,3,head0,greedy0,join0,false);if(!small)run(1,154880,4096,1,head1,greedy1,join1,argc==1);CK(cudaDeviceReset());puts("context_closed=true");}
