#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cmath>
#include "matrix.cuh"
static void ck(cudaError_t e){if(e!=cudaSuccess){fprintf(stderr,"%s\n",cudaGetErrorString(e));exit(1);}}
template<class T> static T *alloc(size_t n){T*p;ck(cudaMalloc(&p,n*sizeof(T)));return p;}
template<class T> static T *upload(const std::vector<T>&v){auto*p=alloc<T>(v.size());ck(cudaMemcpy(p,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice));return p;}
static void launch(int shape,bool next,const int*c,const __nv_bfloat16*x,const __nv_bfloat16*w,__nv_bfloat16*y){
 if(shape==0){if(next)matrix_next_0<<<grid_next_0,128>>>(c,x,w,y);else matrix_old_0<<<grid_old_0,128>>>(c,x,w,y);}
 if(shape==1){if(next)matrix_next_1<<<grid_next_1,128>>>(c,x,w,y);else matrix_old_1<<<grid_old_1,128>>>(c,x,w,y);}
 if(shape==2){if(next)matrix_next_2<<<grid_next_2,128>>>(c,x,w,y);else matrix_old_2<<<grid_old_2,128>>>(c,x,w,y);}
 ck(cudaGetLastError());
}
static void run(int shape,int rows,bool misaligned,bool sanitize){
 const int R=shape?17:64,N=shape?65:4096,K=shape==0?4096:shape==1?33:136;
 std::vector<__nv_bfloat16>x(R*K+1),w(N*K+1);unsigned state=51;
 for(auto* v:{&x,&w})for(auto&a:*v){state=state*1664525u+1013904223u;a=__float2bfloat16_rn((int(state>>16)-32768)*.000031f);}
 auto dx=upload(x),dw=upload(w),a=alloc<__nv_bfloat16>(R*N),b=alloc<__nv_bfloat16>(R*N);auto c=upload<int>({0,1,1,rows,0});
 auto px=dx+(misaligned?1:0),pw=dw+(misaligned?1:0);
 for(int repeat=0;repeat<2;++repeat){
  ck(cudaMemset(a,0x55,R*N*2));ck(cudaMemset(b,0x55,R*N*2));
  launch(shape,false,c,px,pw,a);launch(shape,true,c,px,pw,b);
  std::vector<unsigned short>ha(R*N),hb(R*N);ck(cudaMemcpy(ha.data(),a,R*N*2,cudaMemcpyDeviceToHost));ck(cudaMemcpy(hb.data(),b,R*N*2,cudaMemcpyDeviceToHost));
  int differences=0;for(int i=0;i<R*N;++i)if(ha[i]!=hb[i]){if(differences++<4)fprintf(stderr,"shape=%d rows=%d cell=%d old=%x next=%x\n",shape,rows,i,ha[i],hb[i]);}
  if(differences)exit(2);
  for(int i=(rows>0&&rows<=R?rows:0)*N;i<R*N;++i)if(hb[i]!=0x5555)exit(3);
 }
 printf("correct shape=%d rows=%d misaligned=%d bitwise=1 tail=preserved\n",shape,rows,misaligned);
 if(!sanitize&&shape==0&&rows>0&&rows<=R&&!misaligned){
  for(int i=0;i<8;++i){launch(shape,false,c,px,pw,a);launch(shape,true,c,px,pw,b);}ck(cudaDeviceSynchronize());
  cudaEvent_t start,stop;ck(cudaEventCreate(&start));ck(cudaEventCreate(&stop));
  for(int trial=0;trial<5;++trial)for(int order=0;order<2;++order){bool next=(order^(trial&1))!=0;ck(cudaEventRecord(start));for(int i=0;i<30;++i)launch(shape,next,c,px,pw,next?b:a);ck(cudaEventRecord(stop));ck(cudaEventSynchronize(stop));float ms;ck(cudaEventElapsedTime(&ms,start,stop));printf("timing rows=%d trial=%d next=%d ms=%.9f\n",rows,trial,next,ms/30);}
  ck(cudaEventDestroy(start));ck(cudaEventDestroy(stop));
 }
 for(void*p:{(void*)dx,(void*)dw,(void*)a,(void*)b,(void*)c})ck(cudaFree(p));
}
int main(int argc,char**argv){bool sanitizer=argc>1;
 if(sanitizer){run(0,22,false,true);run(2,17,true,true);}else{
  for(int r:{0,1,3,22,64,65})run(0,r,false,false);
  for(int s:{1,2})for(int r:{0,1,7,16,17,18})for(bool offset:{false,true})run(s,r,offset,false);
 }
 ck(cudaDeviceReset());puts("released=true");
}
