///|
/// Scalar correctness lowering of an explicitly selected BF16 P boundary.
/// Block geometry affects floating-point semantics and is never silently tuned.
fn online_bf16_attention(
  rows : Int,
  heads : Int,
  kv_heads : Int,
  dimension : Int,
  group : Int,
  prefix : String,
  block : Int,
) -> String {
  let storage = block.min(rows)
  (
    $|extern "C" __global__ void \{prefix}_attention(const __nv_bfloat16 *q, const __nv_bfloat16 *k, const __nv_bfloat16 *v, __nv_bfloat16 *out) {
    $|  const int row=blockIdx.x,h=blockIdx.y;
    $|  if(row>=\{rows} || h>=\{heads} || threadIdx.x!=0) return;
    $|  const int kh=h/\{group};
    $|  const unsigned long long qb=((unsigned long long)row*\{heads}+h)*\{dimension};
    $|  float accum[\{dimension}]={0}, scores[\{storage}], maximum=-INFINITY, denominator=0.0f;
    $|  for(int first=0;first<=row;first+=\{block}) {
    $|    const int count=min(\{block},row+1-first);
    $|    float next=maximum;
    $|    for(int slot=0;slot<count;++slot) {
    $|      const unsigned long long kb=((unsigned long long)(first+slot)*\{kv_heads}+kh)*\{dimension};
    $|      float score=0.0f;
    $|      for(int d=0;d<\{dimension};++d) score=__fadd_rn(score,__fmul_rn(__bfloat162float(q[qb+d]),__bfloat162float(k[kb+d])));
    $|      scores[slot]=__fmul_rn(score,rsqrtf(\{dimension}.0f)); next=fmaxf(next,scores[slot]);
    $|    }
    $|    const float alpha=expf(__fsub_rn(maximum,next));
    $|    denominator=__fmul_rn(denominator,alpha);
    $|    for(int d=0;d<\{dimension};++d) accum[d]=__fmul_rn(accum[d],alpha);
    $|    for(int slot=0;slot<count;++slot) {
    $|      const float p=expf(__fsub_rn(scores[slot],next));
    $|      denominator=__fadd_rn(denominator,p);
    $|      const float rounded=__bfloat162float(__float2bfloat16_rn(p));
    $|      const unsigned long long vb=((unsigned long long)(first+slot)*\{kv_heads}+kh)*\{dimension};
    $|      for(int d=0;d<\{dimension};++d) accum[d]=__fmaf_rn(rounded,__bfloat162float(v[vb+d]),accum[d]);
    $|    }
    $|    maximum=next;
    $|  }
    $|  for(int d=0;d<\{dimension};++d) out[qb+d]=__float2bfloat16_rn(__fdiv_rn(accum[d],denominator));
    $|}
  )
}
