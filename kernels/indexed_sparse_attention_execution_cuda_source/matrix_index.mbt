///|
/// Terminal E4M3 fragment ABI. One warp owns eight candidates across sixteen
/// heads, then advances to the next complete head tile. The head reduction is
/// separate from MMA and remains ordered. All shared readers retire before
/// reuse; scales live alongside persistent pooled-key payloads.
fn matrix_index_source(
  plan : @precision.IndexedSparseAttentionPrecision,
  symbol : String,
  ports : String,
  geometry : String,
) -> String {
  let (heads, dimension, pool, _, _) = plan.index_geometry()
  let (queries, _) = plan.capacity()
  let domain = plan.paged_history().unwrap()
  let (page_rows, pages, _, _) = domain.geometry()
  (
    $|#include <cuda_fp8.h>
    $|extern "C" __global__ void \{symbol}\{ports} {
    $| const int live=counts[3];if(live<=0||live>\{queries})return;
    $| const int query=blockIdx.x%live,tile=blockIdx.x/live;
    $| \{geometry}
    $| if(tile*32>=candidates)return;
    $| const int lane=threadIdx.x&31,warp=threadIdx.x/32,g=lane/4,t=lane%4;
    $| __shared__ unsigned char q[\{heads*dimension}],k[32*\{dimension}];
    $| __shared__ float qs[\{heads}],ks[32],dots[\{heads*32}];
    $| __shared__ int valid_keys[32];
    $| for(int head=warp;head<\{heads};head+=4) {
    $|  float maximum=0.0f;for(int c=lane;c<\{dimension};c+=32)maximum=fmaxf(maximum,fabsf(__bfloat162float(queries[((uint64_t)query*\{heads}+head)*\{dimension}+c])));
    $|  for(int stride=16;stride>0;stride>>=1)maximum=fmaxf(maximum,__shfl_xor_sync(0xffffffff,maximum,stride));
    $|  const float scale=__fdiv_rn(fmaxf(maximum,1.0e-12f),448.0f);if(lane==0)qs[head]=scale;
    $|  for(int c=lane;c<\{dimension};c+=32)q[head*\{dimension}+c]=__nv_cvt_float_to_fp8(__fdiv_rn(__bfloat162float(queries[((uint64_t)query*\{heads}+head)*\{dimension}+c]),scale),__NV_SATFINITE,__NV_E4M3);
    $| }
    $| for(int col=warp;col<32;col+=4) {
    $|  const int candidate=tile*32+col,raw=candidate*\{pool},at=first_page+raw/\{page_rows};
    $|  int physical=-1;if(candidate<candidates&&at<end_page){const int page=page_ids[at];if(page>=0&&page<\{pages})physical=page*\{page_rows/pool}+candidate%\{page_rows/pool};}
    $|  const bool valid=physical>=0&&candidate_valid[physical]>0&&candidate_valid[physical]==page_generations[at];
    $|  const unsigned char *encoded=(const unsigned char *)candidate_keys+(uint64_t)(valid?physical:0)*\{dimension+4};
    $|  if(lane==0){valid_keys[col]=valid;ks[col]=valid?*((const float *)(encoded+\{dimension})):0.0f;}
    $|  for(int c=lane;c<\{dimension};c+=32)k[col*\{dimension}+c]=valid?encoded[c]:0;
    $| }
    $| __syncthreads();
    $| for(int head=0;head<\{heads};head+=16) {
    $|  float d[4]={0.0f,0.0f,0.0f,0.0f};
    $|  for(int sub=0;sub<\{dimension};sub+=32) {
    $|   unsigned a[4]={0,0,0,0},b[2]={0,0};
    $|   for(int byte=0;byte<4;++byte) {
    $|    const int c=sub+t*4+byte;
    $|    a[0]|=(unsigned)q[(head+g)*\{dimension}+c]<<(8*byte);
    $|    a[1]|=(unsigned)q[(head+g+8)*\{dimension}+c]<<(8*byte);
    $|    a[2]|=(unsigned)q[(head+g)*\{dimension}+c+16]<<(8*byte);
    $|    a[3]|=(unsigned)q[(head+g+8)*\{dimension}+c+16]<<(8*byte);
    $|    b[0]|=(unsigned)k[(warp*8+g)*\{dimension}+c]<<(8*byte);
    $|    b[1]|=(unsigned)k[(warp*8+g)*\{dimension}+c+16]<<(8*byte);
    $|   }
    $|   asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};" : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]) : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]));
    $|  }
    $|  dots[(head+g)*32+warp*8+t*2]=d[0];dots[(head+g)*32+warp*8+t*2+1]=d[1];
    $|  dots[(head+g+8)*32+warp*8+t*2]=d[2];dots[(head+g+8)*32+warp*8+t*2+1]=d[3];
    $| }
    $| __syncthreads();
    $| if(threadIdx.x<32) {
    $|  const int col=threadIdx.x,candidate=tile*32+col;if(candidate>=candidates)return;
    $|  float score=0.0f;
    $|  for(int head=0;head<\{heads};++head) {
    $|   const float dot=__fmul_rn(dots[head*32+col],__fmul_rn(qs[head],ks[col]));
    $|   const float activated=fmaxf(__fmul_rn(dot,rsqrtf(\{dimension}.0f)),0.0f);
    $|   score=__fadd_rn(score,__fmul_rn(activated,__fmul_rn(__bfloat162float(head_weights[query*\{heads}+head]),rsqrtf(\{heads}.0f))));
    $|  }
    $|  scores[candidate]=score;available[candidate]=valid_keys[col];
    $| }
    $|}
  )
}
