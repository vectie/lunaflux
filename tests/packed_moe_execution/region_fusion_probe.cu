#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "regions.cuh"
static void ck(cudaError_t e){if(e!=cudaSuccess){fprintf(stderr,"%s\n",cudaGetErrorString(e));exit(2);}}
template<class T> T* alloc(size_t n){T*p;ck(cudaMalloc(&p,n*sizeof(T)));ck(cudaMemset(p,0,n*sizeof(T)));return p;}
// Real geometry and ragged geometry use the exact generated source. Repeated
// launch fusion is compared bitwise, not used as a mathematical correctness oracle.
template<bool Tail> void projections(bool once){
  const int K=Tail?tail::K:full::K; const int *width=Tail?tail::widths:full::widths;
  const int *grids=Tail?tail::grids:full::grids;int fg=Tail?tail::fused_grid:full::fused_grid;
  auto*c=alloc<int>(5);auto*x=alloc<__nv_bfloat16>(64*K);
  __nv_bfloat16 *w[6],*a[6],*b[6];
  std::vector<__nv_bfloat16> hx(64*K);for(size_t i=0;i<hx.size();++i)hx[i]=__float2bfloat16_rn((int(i*13%31)-15)/32.f);
  ck(cudaMemcpy(x,hx.data(),hx.size()*2,cudaMemcpyHostToDevice));
  for(int s=0;s<6;++s){w[s]=alloc<__nv_bfloat16>(size_t(width[s])*K);a[s]=alloc<__nv_bfloat16>(64*width[s]);b[s]=alloc<__nv_bfloat16>(64*width[s]);
    std::vector<__nv_bfloat16> hw(size_t(width[s])*K);for(size_t i=0;i<hw.size();++i)hw[i]=__float2bfloat16_rn((int((i*7+s)%23)-11)/64.f);ck(cudaMemcpy(w[s],hw.data(),hw.size()*2,cudaMemcpyHostToDevice));}
  auto run=[&](bool fused){
    if(Tail){if(fused)tail::tail_fused<<<fg,128>>>(c,x,w[0],b[0],w[1],b[1],w[2],b[2],w[3],b[3],w[4],b[4],w[5],b[5]);else {
      tail::tail_separate_0<<<grids[0],128>>>(c,x,w[0],a[0]);tail::tail_separate_1<<<grids[1],128>>>(c,x,w[1],a[1]);tail::tail_separate_2<<<grids[2],128>>>(c,x,w[2],a[2]);tail::tail_separate_3<<<grids[3],128>>>(c,x,w[3],a[3]);tail::tail_separate_4<<<grids[4],128>>>(c,x,w[4],a[4]);tail::tail_separate_5<<<grids[5],128>>>(c,x,w[5],a[5]);}}
    else {if(fused)full::full_fused<<<fg,128>>>(c,x,w[0],b[0],w[1],b[1],w[2],b[2],w[3],b[3],w[4],b[4],w[5],b[5]);else {
      full::full_separate_0<<<grids[0],128>>>(c,x,w[0],a[0]);full::full_separate_1<<<grids[1],128>>>(c,x,w[1],a[1]);full::full_separate_2<<<grids[2],128>>>(c,x,w[2],a[2]);full::full_separate_3<<<grids[3],128>>>(c,x,w[3],a[3]);full::full_separate_4<<<grids[4],128>>>(c,x,w[4],a[4]);full::full_separate_5<<<grids[5],128>>>(c,x,w[5],a[5]);}}
  };
  for(int rows:{0,1,3,16,17,64,65}){if(once&&!Tail&&rows>1&&rows<=64)continue;int counts[5]={0,0,0,rows,0};ck(cudaMemcpy(c,counts,sizeof(counts),cudaMemcpyHostToDevice));
    for(int s=0;s<6;++s){ck(cudaMemset(a[s],0x5a,64*width[s]*2));ck(cudaMemset(b[s],0x5a,64*width[s]*2));}
    run(false);run(true);ck(cudaDeviceSynchronize());
    for(int s=0;s<6;++s){std::vector<unsigned short> ha(64*width[s]),hb(ha.size());ck(cudaMemcpy(ha.data(),a[s],ha.size()*2,cudaMemcpyDeviceToHost));ck(cudaMemcpy(hb.data(),b[s],hb.size()*2,cudaMemcpyDeviceToHost));if(ha!=hb){fprintf(stderr,"region mismatch tail=%d rows=%d segment=%d\n",Tail,rows,s);exit(3);}
      // Independent dyadic inputs/products/sums are exactly representable in
      // F32, making an ordered CPU dot an oracle for both matrix schedules.
      if(rows>0&&rows<=64)for(int r=0;r<rows;++r)for(int col=0;col<width[s];++col){float sum=0;for(int k=0;k<K;++k)sum+=((int((r*K+k)*13%31)-15)/32.f)*((int(((size_t(col)*K+k)*7+s)%23)-11)/64.f);auto expected=__float2bfloat16_rn(sum);unsigned short bits;memcpy(&bits,&expected,2);if(ha[r*width[s]+col]!=bits){fprintf(stderr,"oracle mismatch\n");exit(4);}}
    }
    if(!Tail&&!once&&(rows==1||rows==16||rows==64)){cudaEvent_t begin,end;ck(cudaEventCreate(&begin));ck(cudaEventCreate(&end));
      for(int trial=0;trial<4;++trial)for(int turn=0;turn<2;++turn){bool fused=(trial+turn)%2;for(int warm=0;warm<3;++warm)run(fused);ck(cudaEventRecord(begin));for(int n=0;n<20;++n)run(fused);ck(cudaEventRecord(end));ck(cudaEventSynchronize(end));float ms;ck(cudaEventElapsedTime(&ms,begin,end));printf("rows=%d trial=%d fused=%d chain_ms=%.9f\n",rows,trial,fused,ms/20);}
      ck(cudaEventDestroy(begin));ck(cudaEventDestroy(end));}
    printf("projection tail=%d rows=%d bitwise=true oracle=true\n",Tail,rows);
  }
  for(int s=0;s<6;++s){ck(cudaFree(w[s]));ck(cudaFree(a[s]));ck(cudaFree(b[s]));}ck(cudaFree(x));ck(cudaFree(c));
}
void packing(){constexpr int L=32768,R=4096,M=64;auto*c=alloc<int>(5);auto*a=alloc<unsigned>(M*L);auto*b=alloc<unsigned>(M*R);auto*j=alloc<unsigned>(M*(L+R));
  std::vector<unsigned> ha(M*L),hb(M*R);for(size_t i=0;i<ha.size();++i)ha[i]=unsigned(i)*2654435761u;for(size_t i=0;i<hb.size();++i)hb[i]=~unsigned(i); // Includes arbitrary bit patterns, not floating point arithmetic.
  for(int rows:{0,1,3,17,64,65}){int counts[5]={0,0,0,rows,0};ck(cudaMemcpy(c,counts,sizeof(counts),cudaMemcpyHostToDevice));ck(cudaMemcpy(a,ha.data(),ha.size()*4,cudaMemcpyHostToDevice));ck(cudaMemcpy(b,hb.data(),hb.size()*4,cudaMemcpyHostToDevice));ck(cudaMemset(j,0,M*(L+R)*4));batch_pack<<<M*(L+R)/256,256>>>(c,a,b,j);ck(cudaDeviceSynchronize());
    if(rows>0&&rows<=M){std::vector<unsigned> hj(rows*(L+R));ck(cudaMemcpy(hj.data(),j,hj.size()*4,cudaMemcpyDeviceToHost));for(int i=0;i<rows*(L+R);++i)if(hj[i]!=(i<rows*L?ha[i]:hb[i-rows*L]))exit(5);}
    batch_unpack<<<M*(L+R)/256,256>>>(c,a,b,j);ck(cudaDeviceSynchronize());std::vector<unsigned> aa(ha.size()),bb(hb.size());ck(cudaMemcpy(aa.data(),a,aa.size()*4,cudaMemcpyDeviceToHost));ck(cudaMemcpy(bb.data(),b,bb.size()*4,cudaMemcpyDeviceToHost));if(aa!=ha||bb!=hb)exit(6);
  }ck(cudaFree(c));ck(cudaFree(a));ck(cudaFree(b));ck(cudaFree(j));puts("collective_pack_roundtrip=bitwise");
}
int main(int argc,char**){bool once=argc>1;projections<true>(once);projections<false>(once);packing();ck(cudaDeviceReset());puts("regions passed, CUDA released");}
