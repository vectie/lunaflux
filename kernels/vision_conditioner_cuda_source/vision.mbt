///|
/// Geometry source must be generated from checked grids by the model adapter.
/// It defines prefix_geometry(row,&start,&end,&y,&x,&height,&width).
pub fn position_source(
  rows~ : Int,
  geometry~ : String,
  prefix~ : String,
) -> Bytes raise SourceError {
  validate(prefix, rows, 1152)
  @utf8.encode(
    (
      $|#include <cuda_bf16.h>
      $|#include <math.h>
      $|\{geometry}
      $|extern "C" __global__ void \{prefix}_position(__nv_bfloat16* state,const __nv_bfloat16* table) {
      $| int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=\{rows}*1152)return;
      $| int row=i/1152,c=i%1152,start,end,y,x,h,w; \{prefix}_geometry(row,&start,&end,&y,&x,&h,&w);
      $| float fy=y*47.0f/(h-1),fx=x*47.0f/(w-1);int y0=(int)floorf(fy),x0=(int)floorf(fx),y1=min(y0+1,47),x1=min(x0+1,47);
      $| float dy=fy-y0,dx=fx-x0;
      $| float coeff[4]={(1-dy)*(1-dx),(1-dy)*dx,dy*(1-dx),dy*dx};int index[4]={y0*48+x0,y0*48+x1,y1*48+x0,y1*48+x1};float term[4];
      $| for(int k=0;k<4;++k){float weight=__bfloat162float(__float2bfloat16_rn(coeff[k]));term[k]=__bfloat162float(__float2bfloat16_rn(__bfloat162float(table[index[k]*1152+c])*weight));}
      $| float pos=__bfloat162float(__float2bfloat16_rn(term[0]+term[1]));pos=__bfloat162float(__float2bfloat16_rn(pos+term[2]));pos=__bfloat162float(__float2bfloat16_rn(pos+term[3]));
      $| state[i]=__float2bfloat16_rn(__bfloat162float(state[i])+__bfloat162float(__float2bfloat16_rn(pos)));
      $|}
      $|extern "C" __global__ void \{prefix}_rotary(const __nv_bfloat16* qkv,__nv_bfloat16* out) {
      $| int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=\{rows}*3456)return;
      $| int row=i/3456,lane=i%72,part=i%3456/1152; if(part==2){out[i]=qkv[i];return;}
      $| int start,end,y,x,h,w;\{prefix}_geometry(row,&start,&end,&y,&x,&h,&w);
      $| int half=lane%36,axis=half/18,freq=half%18;float angle=(axis==0?y:x)*powf(10000.0f,-freq/18.0f);
      $| int other=i+(lane<36?36:-36);float rotated=__bfloat162float(qkv[other])*(lane<36?-1.0f:1.0f);
      $| out[i]=__float2bfloat16_rn(__bfloat162float(qkv[i])*cosf(angle)+rotated*sinf(angle));
      $|}
      $|extern "C" __global__ void \{prefix}_attention(const __nv_bfloat16* qkv,__nv_bfloat16* out) {
      $| int row=blockIdx.x,head=blockIdx.y;if(row>=\{rows}||head>=16||threadIdx.x)return;
      $| int start,end,y,x,h,w;\{prefix}_geometry(row,&start,&end,&y,&x,&h,&w);
      $| float accum[72];for(int d=0;d<72;++d)accum[d]=0.0f;float maximum=-INFINITY,denom=0.0f;
      $| for(int key=start;key<end;++key){float score=0.0f;for(int d=0;d<72;++d)score=fmaf(__bfloat162float(qkv[row*3456+head*72+d]),__bfloat162float(qkv[key*3456+1152+head*72+d]),score);
      $| score*=0.1178511301977579f;float next=fmaxf(maximum,score),old=expf(maximum-next),prob=expf(score-next);denom=denom*old+prob;
      $| for(int d=0;d<72;++d)accum[d]=accum[d]*old+prob*__bfloat162float(qkv[key*3456+2304+head*72+d]);maximum=next;}
      $| for(int d=0;d<72;++d)out[row*1152+head*72+d]=__float2bfloat16_rn(accum[d]/denom);
      $|}
    ),
  )
}
