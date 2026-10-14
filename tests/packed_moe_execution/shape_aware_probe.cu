#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include "projection.cuh"
#include "dense.cuh"
#include "text.cuh"
#include "routing.cuh"
static void ck(cudaError_t e){if(e!=cudaSuccess){fprintf(stderr,"%s\n",cudaGetErrorString(e));exit(1);}}
template<class T> static T* alloc(size_t n){T*p;ck(cudaMalloc(&p,n*sizeof(T)));return p;}
template<class T> static T* upload(const std::vector<T>&v){auto p=alloc<T>(v.size());ck(cudaMemcpy(p,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice));return p;}
template<class T> static std::vector<T> read(T*p,size_t n){std::vector<T>v(n);ck(cudaMemcpy(v.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost));return v;}
template<class T> static void exact(T*a,T*b,size_t n,const char*label){auto x=read(a,n),y=read(b,n);if(memcmp(x.data(),y.data(),n*sizeof(T))){fprintf(stderr,"mismatch %s\n",label);exit(2);}}
static std::vector<__nv_bfloat16> pattern(size_t n,int modulus=7){std::vector<__nv_bfloat16>v(n);for(size_t i=0;i<n;++i)v[i]=__float2bfloat16_rn((int(i%modulus)-modulus/2)*.03125f);return v;}
static void projection(int rows){
 const int R=17,K=256,N=512;auto x=pattern(R*K),w=pattern(N*K,11);auto dx=upload(x),dw=upload(w),y=alloc<__nv_bfloat16>(R*N);auto c=upload<int>({0,1,1,rows,0});
 ck(cudaMemset(y,0x55,R*N*2));shape_projection<<<projection_grid,128>>>(c,dx,dw,y);auto z=read(y,R*N);
 if(rows>0&&rows<=R)for(int r=0;r<rows;++r)for(int n=0;n<N;++n){double v=0;for(int k=0;k<K;++k)v+=double(__bfloat162float(x[r*K+k]))*__bfloat162float(w[n*K+k]);auto b=__float2bfloat16_rn(v);if(memcmp(&b,&z[r*N+n],2))exit(3);}
 for(int i=(rows>0&&rows<=R?rows:0)*N;i<R*N;++i){unsigned short bits;memcpy(&bits,&z[i],2);if(bits!=0x5555)exit(4);}
 ck(cudaFree(dx));ck(cudaFree(dw));ck(cudaFree(y));ck(cudaFree(c));printf("projection rows=%d independent_dyadic_oracle=bitwise tail=preserved\n",rows);
}
static void dense(int rows){
 const int R=17,H=64,I=96;auto x=upload(pattern(R*H)),g=upload(pattern(I*H,11)),u=upload(pattern(I*H,13)),d=upload(pattern(H*I,17));auto c=upload<int>({0,1,1,rows,0});
 auto a=alloc<__nv_bfloat16>(R*I),b=alloc<__nv_bfloat16>(R*I),y=alloc<__nv_bfloat16>(R*H),z=alloc<__nv_bfloat16>(R*H);
 for(auto p:{a,b})ck(cudaMemset(p,0x55,R*I*2));for(auto p:{y,z})ck(cudaMemset(p,0x55,R*H*2));
 dense_old_gate_up<<<dense_old_grid0,dense_old_block0>>>(c,x,g,u,a);dense_new_gate_up<<<dense_new_grid0,dense_new_block0>>>(c,x,g,u,b);
 dense_old_down<<<dense_old_grid1,dense_old_block1>>>(c,a,d,y);dense_new_down<<<dense_new_grid1,dense_new_block1>>>(c,b,d,z);
 exact(a,b,R*I,"dense activation");exact(y,z,R*H,"dense output");
 for(void*p:{(void*)x,(void*)g,(void*)u,(void*)d,(void*)c,(void*)a,(void*)b,(void*)y,(void*)z})ck(cudaFree(p));printf("dense rows=%d stages=bitwise\n",rows);
}
static void text(){
 const int V=1024,H=256;auto x=upload(pattern(2*H)),w=upload(pattern(V*H,11));auto n=upload<int>({2}),c=upload<int>({0,1,1,17,0}),s=upload<int>({0,16});auto a=alloc<float>(2*V),b=alloc<float>(2*V);auto ta=alloc<int>(2),tb=alloc<int>(2);
 text_old_head<<<8,256>>>(n,x,w,a);text_new_head<<<256,128>>>(n,x,w,b);exact(a,b,2*V,"head dyadic");
 for(int mode=0;mode<5;++mode){std::vector<float> logits(2*V);for(int i=0;i<2*V;++i)logits[i]=(i%19)-9.f;
  if(mode==1)logits[300]=INFINITY;if(mode==2)logits[V+999]=NAN;if(mode==3)for(auto&v:logits)v=-INFINITY;
  ck(cudaMemcpy(a,logits.data(),logits.size()*4,cudaMemcpyHostToDevice));if(mode==4){int selected[]={-1,17};ck(cudaMemcpy(s,selected,8,cudaMemcpyHostToDevice));}
  text_old_greedy<<<2,1>>>(c,n,s,a,ta);text_new_greedy<<<2,256>>>(c,n,s,a,tb);exact(ta,tb,2,"greedy stable and invalid");
 }
 for(void*p:{(void*)x,(void*)w,(void*)n,(void*)c,(void*)s,(void*)a,(void*)b,(void*)ta,(void*)tb})ck(cudaFree(p));puts("text head=bitwise greedy_ties_nonfinite_invalid=bitwise");
}
static void route(){
 const int R=17,E=288,S=8;auto c=upload<int>({0,1,1,R,0});auto x=alloc<float>(R*E),y=alloc<float>(R*E),a=alloc<float>(R*S),b=alloc<float>(R*S);auto ia=alloc<int>(R*S),ib=alloc<int>(R*S);
 for(int mode=0;mode<4;++mode){std::vector<float> scores(R*E),choice(R*E);for(int i=0;i<R*E;++i){scores[i]=float(i%23)*.03125f;choice[i]=scores[i]+float(i%7)*.03125f;}if(mode==1)choice[100]=NAN;if(mode==2)scores[E+200]=INFINITY;if(mode==3)for(auto&v:choice)v=.5f;
  ck(cudaMemcpy(x,scores.data(),scores.size()*4,cudaMemcpyHostToDevice));ck(cudaMemcpy(y,choice.data(),choice.size()*4,cudaMemcpyHostToDevice));route_old<<<R,1>>>(c,x,y,ia,a);route_new<<<R,128>>>(c,x,y,ib,b);exact(ia,ib,R*S,"router choices");exact(a,b,R*S,"router weights");
 }
 for(void*p:{(void*)c,(void*)x,(void*)y,(void*)a,(void*)b,(void*)ia,(void*)ib})ck(cudaFree(p));puts("grouped selection ties_nonfinite_weights=bitwise");
}
int main(){for(int r:{0,1,7,16,17,18}){projection(r);dense(r);}text();route();ck(cudaDeviceReset());puts("released=true");}
