#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cstring>
#include "matrix.cuh"
static void ck(cudaError_t e) {if(e!=cudaSuccess) {fprintf(stderr,"%s\n",cudaGetErrorString(e));exit(2);}}
template<class T> static T *gpu(const std::vector<T>& v) {T *p;ck(cudaMalloc(&p,v.size()*sizeof(T)));ck(cudaMemcpy(p,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice));return p;}
static float bf(float x) {return __bfloat162float(__float2bfloat16_rn(x));}
static float e2(unsigned x) {const float v[8]={0,.5f,1,1.5f,2,3,4,6};return (x&8)?-v[x&7]:v[x&7];}
static float e4(unsigned x) {__nv_fp8_e4m3 v;v.__x=x;return (float)v;}
static float raw(const unsigned char *p,int r,int k,int width) {return e2((p[(r*width+k)/2]>>(4*(k%2)))&15);}
static float dot(const unsigned char *a,int row,const unsigned char *w,int out,int k,int as,int ag,int ws,int wg) {
  // Independent scalar oracle on the declared packed operands, not the BF16
  // program that preceded activation quantization. Binary block products are
  // accumulated exactly in double before the prescribed F32 publication.
  double sum=0;for(int j=0;j<k;j++) sum+=(double)raw(a,row,j,k)*raw(w,out,j,k)*e4(a[as+row*(k/16)+j/16])*e4(w[ws+out*(k/16)+j/16]);
  float ax,bx;memcpy(&ax,a+ag,4);memcpy(&bx,w+wg,4);return (float)sum*(ax*bx);
}
static unsigned encode(float value) {unsigned sign=std::signbit(value)?8:0;float x=fminf(6,fabsf(value));unsigned best=0;for(unsigned i=1;i<8;i++) {float d=fabsf(x-e2(i)),old=fabsf(x-e2(best));if(d<old||(d==old&&(i%2==0))) best=i;}return best|sign;}
static std::vector<unsigned char> quant(const std::vector<__nv_bfloat16>& x,int rows,int width,int bytes,int so,int go) {
  std::vector<unsigned char> p(bytes);float maximum=0;for(int i=0;i<rows*width;i++) maximum=fmaxf(maximum,fabsf((float)x[i]));float global=maximum==0?1:maximum/2688;memcpy(p.data()+go,&global,4);
  for(int block=0;block<rows*width/16;block++) {float m=0;for(int k=0;k<16;k++) m=fmaxf(m,fabsf((float)x[block*16+k]));float requested=m==0?1:(m/global)/6;unsigned scale=__nv_cvt_float_to_fp8(requested,__NV_SATFINITE,__NV_E4M3);if(!scale) scale=1;p[so+block]=scale;for(int k=0;k<16;k++) {unsigned q=encode(((float)x[block*16+k]/global)/e4(scale));p[(block*16+k)/2]|=q<<(4*(k%2));}}
  return p;
}
int main() {
  std::vector<unsigned char> bank(low_stride*2);
  const int widths[3]={64,64,128},outs[3]={128,128,64};const int so[3]={low_scale0,low_scale1,low_scale2},go[3]={low_global0,low_global1,low_global2};
  for(int e=0;e<2;e++) for(int s=0;s<3;s++) {auto *p=bank.data()+e*low_stride+low_offsets[s];for(int r=0;r<outs[s];r++) {for(int k=0;k<widths[s];k++) p[(r*widths[s]+k)/2]|=((r*3+k*5+e+s)&15)<<(4*(k%2));for(int k=0;k<widths[s]/16;k++) p[so[s]+r*(widths[s]/16)+k]=32+((r+k+e+s)%3)*8;}float global=1.f/32;memcpy(p+go[s],&global,4);}
  std::vector<__nv_bfloat16> x(17*64),act(low_activation_bytes/2);for(int i=0;i<17*64;i++) x[i]=__float2bfloat16_rn(((i*7)%29-14)/32.f);
  std::vector<int> counts(5),indices(34),map{0,1},error(1);std::vector<float> routing(34),output(17*64);std::vector<__nv_bfloat16> weighted(34*64);
  for(int r=0;r<17;r++) {indices[r*2]=r%2;indices[r*2+1]=1-r%2;routing[r*2]=.25f;routing[r*2+1]=.75f;}
  auto *db=gpu(bank),*dxq=gpu(std::vector<unsigned char>(q0_bytes)),*daq=gpu(std::vector<unsigned char>(q1_bytes));auto *dx=gpu(x),*da=gpu(act),*dw=gpu(weighted);auto *dc=gpu(counts),*di=gpu(indices),*dm=gpu(map),*de=gpu(error);auto *ds=gpu(routing),*dy=gpu(output);
  int wrong=0;
  for(int rows: {1,3,16,17,0,1}) {
    counts[3]=rows;ck(cudaMemcpy(dc,counts.data(),20,cudaMemcpyHostToDevice));
    low_group<<<1,128>>>(dc,di,da);low_quantize_0_global<<<1,256>>>(dc,dx,dxq,de);low_quantize_0_pack<<<9,128>>>(dc,dx,dxq,de);
    low_gate_up<<<136,128>>>(dc,dxq,di,dm,db,da);
    low_quantize_1_global<<<1,256>>>(dc,da,daq,de);low_quantize_1_pack<<<34,128>>>(dc,da,daq,de);
    low_down<<<68,128>>>(dc,daq,di,ds,dm,db,dw,da);low_combine<<<5,256>>>(dc,di,dw,dy);ck(cudaGetLastError());ck(cudaDeviceSynchronize());
    if(!rows) continue;
    auto qx=quant(x,rows,64,q0_bytes,q0_scale,q0_global);std::vector<unsigned char> actual(q0_bytes);ck(cudaMemcpy(actual.data(),dxq,q0_bytes,cudaMemcpyDeviceToHost));
    for(int i=0;i<rows*32;i++) if(qx[i]!=actual[i]) wrong++;for(int i=0;i<rows*4;i++) if(qx[q0_scale+i]!=actual[q0_scale+i]) wrong++;if(memcmp(qx.data()+q0_global,actual.data()+q0_global,4)) wrong++;
    std::vector<__nv_bfloat16> reference(34*128);
    for(int job=0;job<rows*2;job++) for(int n=0;n<128;n++) {int e=indices[job];float gate=bf(dot(qx.data(),job/2,bank.data()+e*low_stride+low_offsets[0],n,64,q0_scale,q0_global,so[0],go[0]));float up=bf(dot(qx.data(),job/2,bank.data()+e*low_stride+low_offsets[1],n,64,q0_scale,q0_global,so[1],go[1]));reference[job*128+n]=__float2bfloat16_rn(bf(gate/(1+expf(-gate)))*up);}
    ck(cudaMemcpy(act.data(),da,act.size()*2,cudaMemcpyDeviceToHost));for(int i=0;i<rows*2*128;i++) if(memcmp(&act[i],&reference[i],2)) {if(wrong<8) printf("product mismatch %d %.9g %.9g\n",i,(float)act[i],(float)reference[i]);wrong++;}
    auto qa=quant(reference,rows*2,128,q1_bytes,q1_scale,q1_global);actual.resize(q1_bytes);ck(cudaMemcpy(actual.data(),daq,q1_bytes,cudaMemcpyDeviceToHost));for(int i=0;i<rows*2*64;i++) if(qa[i]!=actual[i]) wrong++;for(int i=0;i<rows*2*8;i++) if(qa[q1_scale+i]!=actual[q1_scale+i]) wrong++;
    ck(cudaMemcpy(output.data(),dy,output.size()*4,cudaMemcpyDeviceToHost));for(int row=0;row<rows;row++) for(int n=0;n<64;n++) {float expected=0;for(int expert=0;expert<2;expert++) for(int slot=0;slot<2;slot++) {int job=row*2+slot;if(indices[job]!=expert) continue;float v=dot(qa.data(),job,bank.data()+expert*low_stride+low_offsets[2],n,128,q1_scale,q1_global,so[2],go[2]);expected+=bf(bf(v)*routing[job]);}if(output[row*64+n]!=expected) {if(wrong<8) printf("output mismatch r%d n%d %.9g %.9g\n",row,n,output[row*64+n],expected);wrong++;}}
    printf("rows=%d mismatches=%d\n",rows,wrong);
  }
  ck(cudaMemcpy(error.data(),de,4,cudaMemcpyDeviceToHost));wrong+=error[0]!=0;
  ck(cudaFree(db));ck(cudaFree(dxq));ck(cudaFree(daq));ck(cudaFree(dx));ck(cudaFree(da));ck(cudaFree(dw));ck(cudaFree(dc));ck(cudaFree(di));ck(cudaFree(dm));ck(cudaFree(de));ck(cudaFree(ds));ck(cudaFree(dy));ck(cudaDeviceReset());printf("mismatches=%d released=true\n",wrong);return wrong?1:0;
}
