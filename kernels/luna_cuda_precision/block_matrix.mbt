///|
/// Private CUDA fragment ABI for raw E4M3 block accumulation followed by a
/// separate scaled fold. IR owns arithmetic; this package owns lane layout.
pub fn e4m3_block_matrix_helper(
  symbol~ : String,
  publication~ : @precision.Float32AffineLaw,
) -> String {
  let law = @precision.ScaledDotReductionLaw::UnscaledBlockMatrixFloat32(
    block_columns=128,
    publication~,
  )
  let raw_fold = scalar_expression(law.publication_expression(), [
    "total[i]", "d[i]", "scale",
  ]) catch {
    _ => abort("invalid scaled publication program")
  }
  (
    $|#include <cuda_bf16.h>
    $|#include <cuda_fp8.h>
    $|#include <math.h>
    $|__device__ __forceinline__ float \{symbol}_ue8(unsigned c) { return c==255U?nanf(""):c==0U?__uint_as_float(0x00400000U):__uint_as_float(c<<23); }
    $|template<int K,int Stride=K> __device__ __forceinline__ void \{symbol}_block_dot(const unsigned char *x,const unsigned char *xs,const unsigned char *w,const unsigned char *ws,int tile,float *total) {
    $|  const int lane=threadIdx.x&31,g=lane/4,t=lane%4;
    $|  total[0]=0.0f;total[1]=0.0f;
    $|  for(int group=0;group<K/128;++group) {
    $|    float d[4]={0.0f,0.0f,0.0f,0.0f};
    $|    for(int sub=0;sub<128;sub+=32) {
    $|      unsigned a[4]={0U,0U,0U,0U},b[2]={0U,0U};
    $|      for(int byte=0;byte<4;++byte) {
    $|        const int col=group*128+sub+t*4+byte;
    $|        if(g==0) { a[0]|=(unsigned)x[col]<<(8*byte); a[2]|=(unsigned)x[col+16]<<(8*byte); }
    $|        b[0]|=(unsigned)w[(unsigned long long)(tile+g)*Stride+col]<<(8*byte);
    $|        b[1]|=(unsigned)w[(unsigned long long)(tile+g)*Stride+col+16]<<(8*byte);
    $|      }
    $|      asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};" : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]) : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]));
    $|    }
    $|    if(g==0) for(int i=0;i<2;++i) {
    $|      const int out=tile+t*2+i;
    $|      const float scale=__fmul_rn(\{symbol}_ue8(xs[group]),\{symbol}_ue8(ws[(out/128)*(Stride/128)+group]));
    $|      total[i]=\{raw_fold};
    $|    }
    $|  }
    $|}
  )
}
