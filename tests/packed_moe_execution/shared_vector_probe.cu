#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "old.cuh"
#include "next.cuh"
static void ck(cudaError_t e){if(e!=cudaSuccess){fprintf(stderr,"%s\n",cudaGetErrorString(e));exit(1);}}
template<class T>T*alloc(size_t n){T*p;ck(cudaMalloc(&p,n*sizeof(T)));return p;}
template<class T>T*upload(const std::vector<T>&v){T*p=alloc<T>(v.size());ck(cudaMemcpy(p,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice));return p;}
__global__ void exhaustive(const unsigned char*p,unsigned short*out){
 int i=(blockIdx.x*blockDim.x+threadIdx.x)*8;if(i>=65536)return;
 uint4 v=next_v0(p,nullptr,nullptr,i/4096,i%4096);
 unsigned short values[8];memcpy(values,&v,16);
 for(int j=0;j<8;++j){auto ref=old_w0(p,nullptr,nullptr,(i+j)/4096,(i+j)%4096);out[(i+j)*2]=__bfloat16_as_ushort(ref);out[(i+j)*2+1]=values[j];}
}
static void all_codes(){
 std::vector<unsigned char>h(65536*2+1);for(int offset:{0,1}){
  for(unsigned i=0;i<65536;++i){h[offset+i*2]=i;h[offset+i*2+1]=i>>8;}
  auto p=upload(h);auto out=alloc<unsigned short>(65536*2);exhaustive<<<64,128>>>(p+offset,out);ck(cudaGetLastError());
  std::vector<unsigned short>v(65536*2);ck(cudaMemcpy(v.data(),out,v.size()*2,cudaMemcpyDeviceToHost));
  for(int i=0;i<65536;++i)if(v[i*2]!=v[i*2+1]){fprintf(stderr,"code=%x scalar=%x vector=%x\n",i,v[i*2],v[i*2+1]);exit(5);}
  ck(cudaFree(p));ck(cudaFree(out));printf("all_bf16_codes=65536 aligned=%d exact=true\n",offset==0);
 }
}
int main(int argc,char**argv){
 bool profile=argc>1&&!strcmp(argv[1],"profile"),san=argc>1&&!strcmp(argv[1],"sanitizer");
 if(!profile)all_codes();
 const int cap=64,k=4096,n=1024;std::vector<__nv_bfloat16>x(cap*k),bank(3ULL*k*n);unsigned state=19;
 for(auto*vec:{&x,&bank})for(auto&a:*vec){state=state*1664525u+1013904223u;a=__float2bfloat16_rn((int(state>>16)-32768)*.000013f);}
 auto dx=upload(x),dw=upload(bank),a=alloc<__nv_bfloat16>(old_activation/2),b=alloc<__nv_bfloat16>(next_activation/2);
 auto ya=alloc<float>(cap*k),yb=alloc<float>(cap*k);auto c=alloc<int>(4),ids=upload(std::vector<int>(cap,0)),map=upload<int>({0});auto weights=upload(std::vector<float>(cap,1));
 auto run=[&](bool next,int part){auto act=next?b:a;auto y=next?yb:ya;
 if(part==0){if(next)next_group<<<next_grid0,next_block0>>>(c,ids,act);else old_group<<<old_grid0,old_block0>>>(c,ids,act);}
 if(part==1){if(next)next_gate_up<<<next_grid1,next_block1>>>(c,dx,ids,map,(unsigned char*)dw,act);else old_gate_up<<<old_grid1,old_block1>>>(c,dx,ids,map,(unsigned char*)dw,act);}
 if(part==2){if(next)next_down<<<next_grid2,next_block2>>>(c,act,ids,weights,map,(unsigned char*)dw,y);else old_down<<<old_grid2,old_block2>>>(c,act,ids,weights,map,(unsigned char*)dw,y);}ck(cudaGetLastError());};
 for(int rows:profile?std::vector<int>{22}:std::vector<int>{0,1,3,22,64,65}){
  int counts[]={0,1,1,rows};ck(cudaMemcpy(c,counts,sizeof(counts),cudaMemcpyHostToDevice));
  ck(cudaMemset(a,0x55,old_activation));ck(cudaMemset(b,0x55,next_activation));ck(cudaMemset(ya,0x55,cap*k*4));ck(cudaMemset(yb,0x55,cap*k*4));
  for(bool next:{false,true})for(int p=0;p<3;++p)run(next,p);
  std::vector<unsigned short>ha(cap*n),hb(cap*n);std::vector<float>oa(cap*k),ob(cap*k);
  ck(cudaMemcpy(ha.data(),a,cap*n*2,cudaMemcpyDeviceToHost));ck(cudaMemcpy(hb.data(),b,cap*n*2,cudaMemcpyDeviceToHost));ck(cudaMemcpy(oa.data(),ya,cap*k*4,cudaMemcpyDeviceToHost));ck(cudaMemcpy(ob.data(),yb,cap*k*4,cudaMemcpyDeviceToHost));
  if(memcmp(ha.data(),hb.data(),cap*n*2)||memcmp(oa.data(),ob.data(),cap*k*4)){fprintf(stderr,"rows=%d mismatch\n",rows);return 4;}
  for(int i=(rows>0&&rows<=cap?rows:0)*n;i<cap*n;++i)if(hb[i]!=0x5555)return 6;
  printf("rows=%d activation/output=bitwise tails=preserved\n",rows);
  if(!san&&!profile&&(rows==1||rows==22||rows==64)){
   cudaEvent_t start,stop;ck(cudaEventCreate(&start));ck(cudaEventCreate(&stop));
   for(int t=0;t<5;++t)for(int order=0;order<2;++order){bool next=(order^(t&1));for(int p=1;p<3;++p){ck(cudaEventRecord(start));for(int i=0;i<20;++i)run(next,p);ck(cudaEventRecord(stop));ck(cudaEventSynchronize(stop));float ms;ck(cudaEventElapsedTime(&ms,start,stop));printf("rows=%d trial=%d next=%d part=%d ms=%.9f\n",rows,t,next,p,ms/20);}}
   ck(cudaEventDestroy(start));ck(cudaEventDestroy(stop));
  }
 }
 for(void*p:{(void*)dx,(void*)dw,(void*)a,(void*)b,(void*)ya,(void*)yb,(void*)c,(void*)ids,(void*)map,(void*)weights})ck(cudaFree(p));
 ck(cudaDeviceReset());puts("released=true");
}
