#include "chunk_delta.cuh"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static void check(cudaError_t e) {
  if(e!=cudaSuccess){fprintf(stderr,"CUDA: %s\n",cudaGetErrorString(e));exit(1);}
}
template<class T> static T* alloc(size_t n){T*p;check(cudaMalloc(&p,n*sizeof(T)));return p;}
template<class T> static void put(T*p,const std::vector<T>&v){check(cudaMemcpy(p,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice));}
template<class T> static std::vector<T> get(T*p,size_t n){std::vector<T>v(n);check(cudaMemcpy(v.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost));return v;}
using Launcher=void(*)(bool,int*,int*,int*,int*,__nv_bfloat16*,__nv_bfloat16*,__nv_bfloat16*,float*,__nv_bfloat16*,float*,__nv_bfloat16*,float*);

struct Probe {
 int h,k,v,rows,seq; size_t sw;
 Launcher launch;
 int *counts,*offsets,*slots,*reset;
 __nv_bfloat16 *q,*key,*value,*beta,*output;
 float *decay,*state,*scratch;
 Probe(int a,int b,int c,int r,int s,size_t w,Launcher f):h(a),k(b),v(c),rows(r),seq(s),sw(w),launch(f){
  counts=alloc<int>(5);offsets=alloc<int>(seq+1);slots=alloc<int>(seq);reset=alloc<int>(seq);
  q=alloc<__nv_bfloat16>(rows*h*k);key=alloc<__nv_bfloat16>(rows*h*k);value=alloc<__nv_bfloat16>(rows*h*v);
  beta=alloc<__nv_bfloat16>(rows*h);output=alloc<__nv_bfloat16>(rows*h*v);
  decay=alloc<float>(rows*h*k);state=alloc<float>(4*h*k*v);scratch=alloc<float>(sw);
 }
 ~Probe(){for(void*p:{(void*)scratch,(void*)state,(void*)decay,(void*)output,(void*)beta,(void*)value,(void*)key,(void*)q,(void*)reset,(void*)slots,(void*)offsets,(void*)counts})check(cudaFree(p));}
 void run(bool chunk){launch(chunk,counts,offsets,slots,reset,q,key,value,decay,beta,state,output,scratch);check(cudaGetLastError());}
 void fixture(bool validate){
  std::vector<__nv_bfloat16>x(rows*h*k),y(x.size()),z(rows*h*v),b(rows*h);
  std::vector<float>g(x.size()),s(4*h*k*v);
  for(size_t i=0;i<x.size();++i){x[i]=__float2bfloat16((int(i%23)-11)*.0625f);y[i]=__float2bfloat16((int(i%17)-8)*.09375f);g[i]=-.001f*(1+i%199);}
  for(size_t i=0;i<z.size();++i)z[i]=__float2bfloat16((int(i%29)-14)*.125f);
  for(size_t i=0;i<b.size();++i)b[i]=__float2bfloat16(float(i%5)*.25f);
  for(size_t i=0;i<s.size();++i)s[i]=(int(i%19)-9)*.00390625f;
  for(int j=0;j<k;++j){x[j]=__float2bfloat16(0);y[j]=__float2bfloat16(0);}
  put(q,x);put(key,y);put(value,z);put(beta,b);put(decay,g);
  std::vector<double> oracle(s.begin(),s.end()); double worst_y=0,worst_s=0;
  for(int step=0;step<(validate?5:1);++step){
   if(step==4){for(size_t j=0;j<g.size();++j)g[j]=(j%3==0)?0.0f:-5.0f;put(decay,g);}
   int active=std::min(seq,rows),live=rows;
   if(step==2){active=1;live=std::max(1,rows-1);} if(step==3){active=0;live=0;}
   std::vector<int>off(seq+1,live),sl(seq,0),rst(seq,0);
   off[0]=0;
   for(int j=0;j<active;++j){off[j]=live*j/active;sl[j]=(j+step)%4;rst[j]=(step==0||step==2||step==4);}
   put(counts,std::vector<int>{0,active,active,live,0});put(offsets,off);put(slots,sl);put(reset,rst);
   const auto before=s;
   std::vector<__nv_bfloat16> expected(rows*h*v,__float2bfloat16(-17));
   if(validate)for(int a=0;a<active;++a){
    const size_t sb0=size_t(sl[a])*h*k*v;
    if(rst[a])std::fill(oracle.begin()+sb0,oracle.begin()+sb0+h*k*v,0.0);
    for(int row=off[a];row<off[a+1];++row)for(int head=0;head<h;++head){
     const size_t qb=(size_t(row)*h+head)*k,vb=(size_t(row)*h+head)*v,sb=sb0+size_t(head)*k*v;
     double qs=0,ks=0;for(int j=0;j<k;++j){double q0=__bfloat162float(x[qb+j]),k0=__bfloat162float(y[qb+j]);qs+=q0*q0;ks+=k0*k0;}
     const double qi=1/std::sqrt(qs+1e-6)/std::sqrt(double(k)),ki=1/std::sqrt(ks+1e-6);
     for(int a0=0;a0<v;++a0){double read=0;
      for(int j=0;j<k;++j){oracle[sb+size_t(j)*v+a0]*=std::exp(double(g[qb+j]));read+=oracle[sb+size_t(j)*v+a0]*__bfloat162float(y[qb+j])*ki;}
      const double delta=(__bfloat162float(z[vb+a0])-read)*__bfloat162float(b[row*h+head]);double result=0;
      for(int j=0;j<k;++j){oracle[sb+size_t(j)*v+a0]+=__bfloat162float(y[qb+j])*ki*delta;result+=oracle[sb+size_t(j)*v+a0]*__bfloat162float(x[qb+j])*qi;}
      expected[vb+a0]=__float2bfloat16(float(result));
     }
    }
   }
   std::vector<float> next;
   if(!validate){put(state,before);break;}
   for(bool chunk:{false,true}){
    put(state,before);put(output,std::vector<__nv_bfloat16>(expected.size(),__float2bfloat16(-17)));
    run(chunk);check(cudaDeviceSynchronize());
    if(!validate)continue;
    auto actual=get(output,expected.size());auto after=get(state,s.size());
    double ey=0,es=0;
    for(size_t j=0;j<actual.size();++j){float a=__bfloat162float(actual[j]);if(!std::isfinite(a)){fprintf(stderr,"nonfinite output\n");exit(2);}ey=std::max(ey,std::fabs(double(a)-__bfloat162float(expected[j])));}
    for(size_t j=0;j<after.size();++j){if(!std::isfinite(after[j])){fprintf(stderr,"nonfinite state\n");exit(2);}es=std::max(es,std::fabs(double(after[j])-oracle[j]));}
    if(ey>.003||es>.003){fprintf(stderr,"oracle mismatch chunk=%d step=%d output=%.9g state=%.9g\n",chunk,step,ey,es);exit(2);}
    for(int slot=0;slot<4;++slot){bool used=false;for(int j=0;j<active;++j)used|=sl[j]==slot;
     if(!used&&memcmp(after.data()+slot*h*k*v,before.data()+slot*h*k*v,h*k*v*sizeof(float))){fprintf(stderr,"unselected slot modified\n");exit(2);}}
    for(size_t j=size_t(live)*h*v;j<actual.size();++j)if(memcmp(&actual[j],&expected[j],2)){fprintf(stderr,"tail modified\n");exit(2);}
    worst_y=std::max(worst_y,ey);worst_s=std::max(worst_s,es);
    if(chunk){next=after;put(state,before);run(true);check(cudaDeviceSynchronize());auto repeated=get(state,s.size());auto y2=get(output,actual.size());
     if(memcmp(repeated.data(),after.data(),after.size()*4)||memcmp(y2.data(),actual.data(),actual.size()*2)){fprintf(stderr,"nondeterministic\n");exit(2);}}
   }
   if(validate)s=next;
  }
  if(validate)printf("oracle_pass=1 output_maxabs=%.9g state_maxabs=%.9g slot_reset_continuation_reorder_tail_idle=passed\n",worst_y,worst_s);
  // Paired timing resets every request to identical state, full live extent.
  std::vector<int>off(seq+1,rows),sl(seq,0),rst(seq,1);off[0]=0;
  put(counts,std::vector<int>{0,1,1,rows,0});put(offsets,off);put(slots,sl);put(reset,rst);
 }
 float time(bool chunk){cudaEvent_t a,b;check(cudaEventCreate(&a));check(cudaEventCreate(&b));check(cudaEventRecord(a));for(int i=0;i<10;++i)run(chunk);check(cudaEventRecord(b));check(cudaEventSynchronize(b));float ms;check(cudaEventElapsedTime(&ms,a,b));check(cudaEventDestroy(b));check(cudaEventDestroy(a));return ms/10;}
};

int main(int argc,char**argv){
 int shape=argc>1?atoi(argv[1]):2;const char*mode=argc>2?argv[2]:"check";
 int h,k,v,r,s;size_t w;Launcher f;
 switch(shape){
 case 0:h=2;k=4;v=6;r=4;s=2;w=scratch0;f=launch0;break;
 case 1:h=4;k=32;v=16;r=8;s=4;w=scratch1;f=launch1;break;
 case 2:h=64;k=128;v=128;r=16;s=2;w=scratch2;f=launch2;break;
 case 3:h=64;k=128;v=128;r=32;s=1;w=scratch3;f=launch3;break;
 case 4:h=64;k=128;v=128;r=1;s=1;w=scratch4;f=launch4;break;
 default:return 2;
 }
 {Probe p(h,k,v,r,s,w,f);const bool profile=strstr(mode,"profile")!=nullptr;p.fixture(!profile);
  if(profile){p.run(strcmp(mode,"profile-new")==0);check(cudaDeviceSynchronize());}
  if(strcmp(mode,"timing")==0){p.run(false);p.run(true);check(cudaDeviceSynchronize());for(int i=0;i<5;++i){float old,fresh;if(i%2){fresh=p.time(true);old=p.time(false);}else{old=p.time(false);fresh=p.time(true);}printf("shape=%d rows=%d trial=%d ordered_ms=%.9g chunk_ms=%.9g\n",shape,r,i,old,fresh);}}
 }
 check(cudaDeviceReset());printf("released=true\n");
}
