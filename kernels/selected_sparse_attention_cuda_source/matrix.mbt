///|
/// Terminal NVIDIA fragment layout. Only instruction row zero is live; columns
/// are independent selected keys (QK) or value components (PV). Different
/// queries/heads never share a selection or KV address by assumption.
fn matrix_bodies(
  qk : Int,
  values : Int,
  kv_heads : Int,
  symbol : String,
) -> (String, String, String) raise {
  guard @physical.SelectedMatrixFold::new(
      qk_dimension=qk,
      value_dimension=values,
      columns=8,
      reduction=16,
    )
    is Some(plan) else {
    raise @precision.PrecisionPlanError::NoImplementation
  }
  let (columns, reduction) = plan.geometry()
  let helper =
    $|__device__ __forceinline__ unsigned \{symbol}_pair(__nv_bfloat16 a,__nv_bfloat16 b) { return (unsigned)__bfloat16_as_ushort(a)|((unsigned)__bfloat16_as_ushort(b)<<16); }
    $|__device__ __forceinline__ void \{symbol}_mma(unsigned a0,unsigned a2,unsigned b0,unsigned b1,float *d) {
    $| asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%5}, {%7,%8}, {%0,%1,%2,%3};" : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]) : "r"(a0),"r"(0U),"r"(a2),"r"(b0),"r"(b1));
    $|}
  let scores =
    $|// Explicit tiled-F32 numerical program; BF16 inputs, F32 MMA accumulation.
    $|const int lane=worker&31,warp=worker/32,g=lane/4,t=lane%4;
    $|for(int tile=warp*\{columns};tile<visible;tile+=(blockDim.x/32)*\{columns}) {
    $| float d[4]={0.0f,0.0f,0.0f,0.0f};
    $| const int item=tile+g;
    $| const unsigned long long kbase=item<visible?((unsigned long long)retained[item]*\{kv_heads}+kv_head)*\{qk}ULL:0ULL;
    $| for(int base=0;base<\{qk};base+=\{reduction}) {
    $|  const int c=base+t*2;
    $|  unsigned a0=g==0?\{symbol}_pair(queries[qbase+c],queries[qbase+c+1]):0U;
    $|  unsigned a2=g==0?\{symbol}_pair(queries[qbase+c+8],queries[qbase+c+9]):0U;
    $|  unsigned b0=item<visible?\{symbol}_pair(keys[kbase+c],keys[kbase+c+1]):0U;
    $|  unsigned b1=item<visible?\{symbol}_pair(keys[kbase+c+8],keys[kbase+c+9]):0U;
    $|  \{symbol}_mma(a0,a2,b0,b1,d);
    $| }
    $| if(g==0)for(int i=0;i<2;++i)if(tile+t*2+i<visible)scores[tile+t*2+i]=__fmul_rn(d[i],scale);
    $|}
  let output =
    $|for(int column=warp*\{columns};column<\{values};column+=(blockDim.x/32)*\{columns}) {
    $| float d[4]={0.0f,0.0f,0.0f,0.0f};
    $| for(int base=0;base<visible;base+=\{reduction}) {
    $|  const int a=base+t*2,b=a+1,c=a+8,e=a+9;
    $|  const __nv_bfloat16 zero=__float2bfloat16_rn(0.0f);
    $|  unsigned a0=g==0?\{symbol}_pair(a<visible?__float2bfloat16_rn(scores[a]):zero,b<visible?__float2bfloat16_rn(scores[b]):zero):0U;
    $|  unsigned a2=g==0?\{symbol}_pair(c<visible?__float2bfloat16_rn(scores[c]):zero,e<visible?__float2bfloat16_rn(scores[e]):zero):0U;
    $|  const int out=column+g;
    $|  unsigned b0=\{symbol}_pair(a<visible?values[((unsigned long long)retained[a]*\{kv_heads}+kv_head)*\{values}ULL+out]:zero,b<visible?values[((unsigned long long)retained[b]*\{kv_heads}+kv_head)*\{values}ULL+out]:zero);
    $|  unsigned b1=\{symbol}_pair(c<visible?values[((unsigned long long)retained[c]*\{kv_heads}+kv_head)*\{values}ULL+out]:zero,e<visible?values[((unsigned long long)retained[e]*\{kv_heads}+kv_head)*\{values}ULL+out]:zero);
    $|  \{symbol}_mma(a0,a2,b0,b1,d);
    $| }
    $| if(g==0)for(int i=0;i<2;++i)output[outbase+column+t*2+i]=__float2bfloat16_rn(d[i]);
    $|}
  (helper, scores, output)
}
