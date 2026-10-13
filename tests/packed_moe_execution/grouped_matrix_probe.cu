#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include "grouped.cuh"
static void ck(cudaError_t e){if(e!=cudaSuccess){fprintf(stderr,"%s\n",cudaGetErrorString(e));exit(1);}}
template<class T>static T* alloc(size_t n){T*p;ck(cudaMalloc(&p,n*sizeof(T)));return p;}
static float bf(float v){return __bfloat162float(__float2bfloat16_rn(v));}
static float weight(int local,int stage,int out,int k){return float(((local+stage+out*3+k)%5)-2)*.5f;}
static unsigned nibble(float v){return v<0?unsigned(-v*2)|8:unsigned(v*2);}
__global__ void decode_codes(float*out){int i=threadIdx.x;if(i<16)out[i]=lf_precision_e2(i);}
static void check_decode(){
 auto p=alloc<float>(16);decode_codes<<<1,32>>>(p);float actual[16],expected[16];
 const float values[]={0.f,.5f,1.f,1.5f,2.f,3.f,4.f,6.f};
 for(int i=0;i<16;++i)expected[i]=(i&8)?-values[i&7]:values[i&7];
 ck(cudaMemcpy(actual,p,sizeof(actual),cudaMemcpyDeviceToHost));
 if(memcmp(actual,expected,sizeof(actual))){fprintf(stderr,"E2M1 code mismatch including signed zero\n");exit(6);}
 ck(cudaFree(p));puts("e2m1_all16_codes=bitwise");
}
__global__ void vector_decode_check(const unsigned char*p,const unsigned char*s,const float*g,__nv_bfloat16*out){
 int lane=threadIdx.x;if(lane>=8)return;
 uint4 values=nv_matrix_v0(p,s,g,0,lane*8);reinterpret_cast<uint4*>(out)[lane]=values;
 for(int j=0;j<8;++j)out[64+lane*8+j]=nv_matrix_w0(p,s,g,0,lane*8+j);
}
static void check_vector_decode(){
 auto p=alloc<unsigned char>(33),s=alloc<unsigned char>(4);auto g=alloc<float>(1);auto out=alloc<__nv_bfloat16>(128);
 unsigned char payload[33];for(int i=0;i<33;++i)payload[i]=unsigned((i*2)%16)|unsigned(((i*2+1)%16)<<4);
 for(int offset:{0,1})for(unsigned scale:{1,2,0x38,0x7e})for(float global:{.0001f,.33f,1.f,3.42f,1.e30f}){
  ck(cudaMemcpy(p,payload,33,cudaMemcpyHostToDevice));ck(cudaMemset(s,scale,4));ck(cudaMemcpy(g,&global,4,cudaMemcpyHostToDevice));
  vector_decode_check<<<1,32>>>(p+offset,s,g,out);__nv_bfloat16 actual[128];ck(cudaMemcpy(actual,out,sizeof(actual),cudaMemcpyDeviceToHost));
  if(memcmp(actual,actual+64,128)){fprintf(stderr,"vector decode mismatch offset=%d scale=%u global=%g\n",offset,scale,global);exit(7);}
 }
 ck(cudaFree(p));ck(cudaFree(s));ck(cudaFree(g));ck(cudaFree(out));puts("vector_fp4_scale_rounding_unaligned=bitwise");
}
static void run(bool packed,int rows){
 const int H=64,I=32,R=17,S=2,J=R*S;
 size_t stride=packed?nv_matrix_stride:bf_matrix_stride;
 const size_t* offsets=packed?nv_matrix_offsets:bf_matrix_offsets;
 std::vector<unsigned char>bank(stride*2,0);
 for(int e=0;e<2;++e)for(int stage=0;stage<3;++stage){int n=stage==2?H:I,k=stage==2?I:H;unsigned char*p=bank.data()+e*stride+offsets[stage];
  if(packed){for(int o=0;o<n;++o)for(int j=0;j<k;j+=2)p[(o*k+j)/2]=nibble(weight(e,stage,o,j))|(nibble(weight(e,stage,o,j+1))<<4);
   memset(p+n*k/2,0x38,n*k/16);float scale=1;size_t g=(n*k/2+n*k/16+3)/4*4;memcpy(p+g,&scale,4);
  }else{for(int o=0;o<n;++o)for(int j=0;j<k;++j){auto w=__float2bfloat16_rn(weight(e,stage,o,j));memcpy(p+(o*k+j)*2,&w,2);}}
 }
 std::vector<__nv_bfloat16>x(R*H);for(int i=0;i<R*H;++i)x[i]=__float2bfloat16_rn((i%13-6)*.03125f);
 std::vector<int>ix(J);std::vector<float>sw(J);int em[]={-1,1,-1,-1,-1,0,-1};
 for(int r=0;r<R;++r){ix[r*2]=r%3==0?1:5;ix[r*2+1]=r%3==0?5:(r%3==1?3:99);sw[r*2]=.25f;sw[r*2+1]=.75f;}
 auto db=alloc<unsigned char>(bank.size());auto dx=alloc<__nv_bfloat16>(x.size());auto di=alloc<int>(J);auto ds=alloc<float>(J);auto dm=alloc<int>(7);auto dc=alloc<int>(5);
 auto a=alloc<__nv_bfloat16>(J*I),b=alloc<__nv_bfloat16>(J*H);auto y=alloc<float>(R*H);
 auto am=alloc<__nv_bfloat16>(4492/2),bm=alloc<__nv_bfloat16>(J*H);auto ym=alloc<float>(R*H);
 int counts[]={0,1,1,rows,0};
 ck(cudaMemcpy(db,bank.data(),bank.size(),cudaMemcpyHostToDevice));ck(cudaMemcpy(dx,x.data(),x.size()*2,cudaMemcpyHostToDevice));ck(cudaMemcpy(di,ix.data(),J*4,cudaMemcpyHostToDevice));ck(cudaMemcpy(ds,sw.data(),J*4,cudaMemcpyHostToDevice));ck(cudaMemcpy(dm,em,sizeof(em),cudaMemcpyHostToDevice));ck(cudaMemcpy(dc,counts,sizeof(counts),cudaMemcpyHostToDevice));
 ck(cudaMemset(a,0x55,J*I*2));ck(cudaMemset(am,0x55,4492));ck(cudaMemset(b,0x55,J*H*2));ck(cudaMemset(bm,0x55,J*H*2));ck(cudaMemset(y,0x55,R*H*4));ck(cudaMemset(ym,0x55,R*H*4));
 auto launch=[&](bool matrix){
  if(packed){if(matrix){nv_matrix_group<<<1,128>>>(dc,di,am);nv_matrix_gate_up<<<34,128>>>(dc,dx,di,dm,db,am);nv_matrix_down<<<34,128>>>(dc,am,di,ds,dm,db,bm);nv_matrix_combine<<<5,256>>>(dc,di,bm,ym);}
   else{nv_scalar_gate_up<<<272,128>>>(dc,dx,di,dm,db,a);nv_scalar_down<<<544,128>>>(dc,a,di,ds,dm,db,b);nv_scalar_combine<<<5,256>>>(dc,di,b,y);}}
  else{if(matrix){bf_matrix_group<<<1,128>>>(dc,di,am);bf_matrix_gate_up<<<34,128>>>(dc,dx,di,dm,db,am);bf_matrix_down<<<34,128>>>(dc,am,di,ds,dm,db,bm);bf_matrix_combine<<<5,256>>>(dc,di,bm,ym);}
   else{bf_scalar_gate_up<<<272,128>>>(dc,dx,di,dm,db,a);bf_scalar_down<<<544,128>>>(dc,a,di,ds,dm,db,b);bf_scalar_combine<<<5,256>>>(dc,di,b,y);}}
  ck(cudaGetLastError());
 };
 launch(false);launch(true);ck(cudaDeviceSynchronize());
 std::vector<__nv_bfloat16>ha(J*I),hm(J*I),hb(J*H),hbm(J*H);std::vector<float>hy(R*H),hym(R*H);
 ck(cudaMemcpy(ha.data(),a,J*I*2,cudaMemcpyDeviceToHost));ck(cudaMemcpy(hm.data(),am,J*I*2,cudaMemcpyDeviceToHost));ck(cudaMemcpy(hb.data(),b,J*H*2,cudaMemcpyDeviceToHost));ck(cudaMemcpy(hbm.data(),bm,J*H*2,cudaMemcpyDeviceToHost));ck(cudaMemcpy(hy.data(),y,R*H*4,cudaMemcpyDeviceToHost));ck(cudaMemcpy(hym.data(),ym,R*H*4,cudaMemcpyDeviceToHost));
 if(memcmp(ha.data(),hm.data(),J*I*2)||memcmp(hb.data(),hbm.data(),J*H*2)||memcmp(hy.data(),hym.data(),R*H*4)){fprintf(stderr,"stage mismatch packed=%d rows=%d\n",packed,rows);exit(3);}
 // Independent scalar mathematical fixture: dyadic products are exact; retain
 // the declared BF16 activation/down rounds and expert-ID publication order.
 if(rows>0&&rows<=R)for(int r=0;r<rows;++r){std::vector<float>acc(H,0);
  for(int expert=0;expert<7;++expert)for(int s=0;s<S;++s)if(ix[r*S+s]==expert&&em[expert]>=0){int e=em[expert];std::vector<float>act(I);
   for(int o=0;o<I;++o){double gate=0,up=0;for(int k=0;k<H;++k){gate+=double(__bfloat162float(x[r*H+k]))*weight(e,0,o,k);up+=double(__bfloat162float(x[r*H+k]))*weight(e,1,o,k);}float g=bf(std::min(bf(gate),10.f)),u=bf(std::max(-10.f,std::min(bf(up),10.f)));act[o]=bf(bf(g/(1.f+expf(-g)))*u);}
   for(int o=0;o<H;++o){double sum=0;for(int k=0;k<I;++k)sum+=double(act[k])*weight(e,2,o,k);acc[o]+=bf(bf(sum)*sw[r*S+s]);}
  }
  for(int o=0;o<H;++o)if(acc[o]!=hym[r*H+o]){fprintf(stderr,"oracle mismatch row=%d col=%d %.9g %.9g\n",r,o,acc[o],hym[r*H+o]);exit(4);}
 }
 launch(true);ck(cudaDeviceSynchronize());std::vector<float>repeat(R*H);ck(cudaMemcpy(repeat.data(),ym,R*H*4,cudaMemcpyDeviceToHost));if(memcmp(repeat.data(),hym.data(),R*H*4))exit(5);
 for(void*p:{(void*)db,(void*)dx,(void*)di,(void*)ds,(void*)dm,(void*)dc,(void*)a,(void*)am,(void*)b,(void*)bm,(void*)y,(void*)ym})ck(cudaFree(p));
 printf("format=%s rows=%d stages=bitwise oracle=pass repeat=bitwise\n",packed?"nvfp4":"bf16",rows);
}
int main(){check_decode();check_vector_decode();for(int rows:{0,1,7,16,17,18})for(bool packed:{false,true})run(packed,rows);ck(cudaDeviceReset());puts("released=true");}
