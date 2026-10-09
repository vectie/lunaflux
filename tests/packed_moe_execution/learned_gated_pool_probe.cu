#include "pool_kernels.cuh"
#include <cuda_runtime.h>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <cmath>
#define CK(expr) do { const cudaError_t e=(expr); if(e!=cudaSuccess){std::fprintf(stderr,"%s: %s\n",#expr,cudaGetErrorString(e));std::exit(2);} } while(0)
template<class T> T *device(const std::vector<T>& host) { T *p; CK(cudaMalloc(&p,host.size()*sizeof(T))); CK(cudaMemcpy(p,host.data(),host.size()*sizeof(T),cudaMemcpyHostToDevice)); return p; }
template<class T> void read(std::vector<T>& host,T *p) { CK(cudaMemcpy(host.data(),p,host.size()*sizeof(T),cudaMemcpyDeviceToHost)); }
float f(__nv_bfloat16 x) { return __bfloat162float(x); }
__nv_bfloat16 b(float x) { return __float2bfloat16_rn(x); }
void test(int ratio,bool overlap) {
  const int width=16,hidden=8,rows=5,channels=(overlap?2:1)*width,srows=(overlap?2:1)*ratio,maxout=(rows+ratio-1)/ratio;
  std::vector<__nv_bfloat16> x(rows*hidden),wkv(channels*hidden),wg(channels*hidden),norm(width),pooled(maxout*width,b(19)),output=pooled;
  std::vector<float> ape(ratio*channels),kv(rows*channels),score=kv,state(srows*channels,71),scores=state,expected(state.size(),0),es(expected.size(),-INFINITY);
  std::vector<int32_t> counts(5),positions(rows),reset(1),history(2),append(6),oc(5),op(maxout);
  for(int c=0;c<channels;c++) for(int j=0;j<hidden;j++){wkv[c*hidden+j]=b(float((c*hidden+j)%17-8)/64);wg[c*hidden+j]=b(float((c*hidden+j)%11-5)/32);}
  for(int c=0;c<width;c++)norm[c]=b(1+float(c%3)/16);
  for(int i=0;i<ratio*channels;i++)ape[i]=float(i%7-3)/16;
  auto dx=device(x),dwkv=device(wkv),dwg=device(wg),dn=device(norm),dp=device(pooled),dout=device(output);
  auto da=device(ape),dk=device(kv),ds=device(score),dst=device(state),dss=device(scores);
  auto dc=device(counts),dpos=device(positions),dr=device(reset),dh=device(history),dap=device(append),doc=device(oc),dop=device(op);
  int base=0,total=0;float largest=0;
  const std::vector<int> chunks=overlap?std::vector<int>{1,3,5,2,4,0,5,1}:std::vector<int>(27,5);
  for(int step=0;step<(int)chunks.size();step++) {
    int live=chunks[step]; reset[0]=(step==0); counts[2]=live?1:0;counts[3]=live;
    for(int r=0;r<live;r++){positions[r]=base+r;for(int c=0;c<hidden;c++)x[r*hidden+c]=b(float(((base+r)*hidden+c)%13-6)/8);}
    CK(cudaMemcpy(dx,x.data(),x.size()*2,cudaMemcpyHostToDevice));CK(cudaMemcpy(dc,counts.data(),20,cudaMemcpyHostToDevice));CK(cudaMemcpy(dpos,positions.data(),positions.size()*4,cudaMemcpyHostToDevice));CK(cudaMemcpy(dr,reset.data(),4,cudaMemcpyHostToDevice));
    if(overlap){pool4_reserve<<<1,1>>>(dc,dpos,dr,dh,dap,doc,dop);pool4_project<<<dim3(rows,1),256>>>(dap,dx,dwkv,dwg,dk,ds);pool4_pool<<<1,256>>>(dap,dk,ds,da,dst,dss,dp);pool4_norm<<<maxout,256>>>(dap,dp,dn,dout);pool4_publish<<<1,1>>>(dap,dh);}
    else{pool128_reserve<<<1,1>>>(dc,dpos,dr,dh,dap,doc,dop);pool128_project<<<dim3(rows,1),256>>>(dap,dx,dwkv,dwg,dk,ds);pool128_pool<<<1,256>>>(dap,dk,ds,da,dst,dss,dp);pool128_norm<<<maxout,256>>>(dap,dp,dn,dout);pool128_publish<<<1,1>>>(dap,dh);}
    CK(cudaGetLastError());CK(cudaDeviceSynchronize());
    std::vector<__nv_bfloat16> want(maxout*width);int completed=0;
    for(int r=0;r<live;r++) {
      const int position=base+r,local=position%ratio;
      for(int c=0;c<channels;c++){float v=0,s=0;for(int j=0;j<hidden;j++){v+=f(x[r*hidden+j])*f(wkv[c*hidden+j]);s+=f(x[r*hidden+j])*f(wg[c*hidden+j]);}const int index=((overlap?ratio:0)+local)*channels+c;expected[index]=v;es[index]=s+ape[local*channels+c];}
      if((position+1)%ratio)continue;
      for(int c=0;c<width;c++){
        float maximum=-INFINITY,denominator=0,sum=0;
        for(int slot=0;slot<srows;slot++){const int index=slot*channels+(overlap&&slot>=ratio?width:0)+c;maximum=std::max(maximum,es[index]);}
        for(int slot=0;slot<srows;slot++){const int index=slot*channels+(overlap&&slot>=ratio?width:0)+c;denominator+=std::exp(es[index]-maximum);}
        for(int slot=0;slot<srows;slot++){const int index=slot*channels+(overlap&&slot>=ratio?width:0)+c;sum+=expected[index]*(std::exp(es[index]-maximum)/denominator);}
        want[completed*width+c]=b(sum);
      }
      if(overlap)for(int i=0;i<ratio*channels;i++){expected[i]=expected[ratio*channels+i];es[i]=es[ratio*channels+i];}
      completed++;
    }
    read(pooled,dp);read(output,dout);read(state,dst);read(scores,dss);read(history,dh);read(append,dap);read(oc,doc);read(op,dop);
    if(history[0]!=base+live||history[1]||append[2]||oc[3]!=completed){std::fprintf(stderr,"metadata failure\n");std::exit(3);}
    for(int r=0;r<completed;r++){
      if(op[r]!=(base/ratio+r)*ratio){std::fprintf(stderr,"position failure\n");std::exit(3);}
      float sum=0;for(int c=0;c<width;c++){float v=f(want[r*width+c]);sum+=v*v;}
      const float inverse=1/std::sqrt(sum/width+1.0e-6f);
      for(int c=0;c<width;c++){float expected_output=f(b(f(want[r*width+c])*inverse*f(norm[c])));float error=std::fabs(f(output[r*width+c])-expected_output);largest=std::max(largest,error);if(error>0.02f+0.01f*std::fabs(expected_output)){std::fprintf(stderr,"numeric failure %g\n",error);std::exit(4);}}
    }
    for(size_t i=0;i<state.size();i++){if(state[i]!=expected[i]||scores[i]!=es[i]){std::fprintf(stderr,"state failure %zu %g/%g %g/%g\n",i,state[i],expected[i],scores[i],es[i]);std::exit(5);}}
    base+=live;total+=completed;
  }
  // A positional gap must poison the descriptor without changing learned state.
  auto before=state;counts[2]=1;counts[3]=1;positions[0]=base+1;reset[0]=0;
  CK(cudaMemcpy(dc,counts.data(),20,cudaMemcpyHostToDevice));CK(cudaMemcpy(dpos,positions.data(),positions.size()*4,cudaMemcpyHostToDevice));CK(cudaMemcpy(dr,reset.data(),4,cudaMemcpyHostToDevice));
  if(overlap){pool4_reserve<<<1,1>>>(dc,dpos,dr,dh,dap,doc,dop);pool4_pool<<<1,256>>>(dap,dk,ds,da,dst,dss,dp);pool4_publish<<<1,1>>>(dap,dh);}
  else{pool128_reserve<<<1,1>>>(dc,dpos,dr,dh,dap,doc,dop);pool128_pool<<<1,256>>>(dap,dk,ds,da,dst,dss,dp);pool128_publish<<<1,1>>>(dap,dh);}
  CK(cudaDeviceSynchronize());read(state,dst);read(history,dh);if(state!=before||history[0]!=base||history[1]!=1){std::fprintf(stderr,"gap failure\n");std::exit(6);}
  for(void*p:{(void*)dx,(void*)dwkv,(void*)dwg,(void*)dn,(void*)dp,(void*)dout,(void*)da,(void*)dk,(void*)ds,(void*)dst,(void*)dss,(void*)dc,(void*)dpos,(void*)dr,(void*)dh,(void*)dap,(void*)doc,(void*)dop})CK(cudaFree(p));
  std::printf("ratio=%d overlap=%d tokens=%d compressed=%d largest_output_error=%.9g passed\n",ratio,overlap,base,total,largest);
}
int main(){test(4,true);test(128,false);return 0;}
