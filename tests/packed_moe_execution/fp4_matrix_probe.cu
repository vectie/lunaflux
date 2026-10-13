#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include "matrix.cuh"
static void ck(cudaError_t e) { if(e!=cudaSuccess) { fprintf(stderr,"%s\n",cudaGetErrorString(e)); exit(2); } }
// Nonuniform codes/scales expose every row, column, scale group and sign bit.
__global__ void probe(const unsigned *a,const unsigned *b,const unsigned *sa,const unsigned *sb,float *out) {
  int lane=threadIdx.x,g=lane/4,t=lane%4;
  float d[4]={0,0,0,0};
  native_fp4_mma(d,a[g*8+t],a[(g+8)*8+t],a[g*8+t+4],a[(g+8)*8+t+4],b[g*8+t],b[g*8+t+4],sa[g+(t&1)*8],sb[g]);
  out[g*8+t*2]=d[0];out[g*8+t*2+1]=d[1];out[(g+8)*8+t*2]=d[2];out[(g+8)*8+t*2+1]=d[3];
}
static float e2(unsigned x) { const float v[8]={0,.5f,1,1.5f,2,3,4,6}; return (x&8)?-v[x&7]:v[x&7]; }
int main() {
  unsigned a[128]={},b[64]={},sa[16]={},sb[8]={};float y[128];
  for(int r=0;r<16;r++) { for(int k=0;k<64;k++) a[r*8+k/8]|=((r*3+k*7+1)&15)<<(4*(k%8));for(int k=0;k<4;k++) sa[r]|=(unsigned)(48+((r+k)%4)*8)<<(8*k); }
  for(int r=0;r<8;r++) { for(int k=0;k<64;k++) b[r*8+k/8]|=((r*5+k*3+2)&15)<<(4*(k%8));for(int k=0;k<4;k++) sb[r]|=(unsigned)(40+((r*2+k)%4)*8)<<(8*k); }
  unsigned *da,*db,*dsa,*dsb;float *dy;ck(cudaMalloc(&da,sizeof(a)));ck(cudaMalloc(&db,sizeof(b)));ck(cudaMalloc(&dsa,sizeof(sa)));ck(cudaMalloc(&dsb,sizeof(sb)));ck(cudaMalloc(&dy,sizeof(y)));
  ck(cudaMemcpy(da,a,sizeof(a),cudaMemcpyHostToDevice));ck(cudaMemcpy(db,b,sizeof(b),cudaMemcpyHostToDevice));ck(cudaMemcpy(dsa,sa,sizeof(sa),cudaMemcpyHostToDevice));ck(cudaMemcpy(dsb,sb,sizeof(sb),cudaMemcpyHostToDevice));
  probe<<<1,32>>>(da,db,dsa,dsb,dy);ck(cudaGetLastError());ck(cudaMemcpy(y,dy,sizeof(y),cudaMemcpyDeviceToHost));
  int wrong=0;for(int r=0;r<16;r++) for(int n=0;n<8;n++) { float sum=0;for(int k=0;k<64;k++) {__nv_fp8_e4m3 as,bs;as.__x=(sa[r]>>(8*(k/16)))&255;bs.__x=(sb[n]>>(8*(k/16)))&255;sum+=e2(a[r*8+k/8]>>(4*(k%8)))*e2(b[n*8+k/8]>>(4*(k%8)))*(float)as*(float)bs;}if(y[r*8+n]!=sum) {if(wrong<8) printf("mismatch %d %d actual %.9g expected %.9g\n",r,n,y[r*8+n],sum);wrong++;}}
  ck(cudaFree(da));ck(cudaFree(db));ck(cudaFree(dsa));ck(cudaFree(dsb));ck(cudaFree(dy));ck(cudaDeviceReset());printf("values=128 mismatches=%d released=true\n",wrong);return wrong?1:0;
}
