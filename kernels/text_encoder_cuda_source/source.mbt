///|
pub(all) suberror TextEncoderSourceError {
  InvalidShape
  InvalidSymbol
  InvalidEpsilon
} derive(Eq)

///|
fn symbol_ok(s : String) -> Bool {
  !s.is_empty() &&
  s.length() <= 96 &&
  s[0] is ('a'..='z' | 'A'..='Z' | '_') &&
  s.all(c => c is ('a'..='z' | 'A'..='Z' | '0'..='9' | '_'))
}

///|
/// Correctness implementation for a single unpadded causal sequence. Cos/sin
/// are precomputed BF16 split-half rotary tables [rows,dimension]. No KV cache.
/// Q/K normalization rounds the normalized value before multiplying BF16 scale.
/// Numerical parity and fast tiled lowering require physical qualification.
pub fn attention_source(
  rows~ : Int,
  query_heads~ : Int,
  kv_heads~ : Int,
  dimension~ : Int,
  epsilon~ : Double,
  prefix~ : String,
) -> Bytes raise TextEncoderSourceError {
  guard rows > 0 &&
    rows <= 262144 &&
    query_heads > 0 &&
    kv_heads > 0 &&
    query_heads <= 1024 &&
    kv_heads <= query_heads &&
    query_heads % kv_heads == 0 &&
    dimension > 0 &&
    dimension <= 256 &&
    dimension % 2 == 0 else {
    raise InvalidShape
  }
  guard symbol_ok(prefix) else { raise InvalidSymbol }
  guard epsilon > 0.0 && epsilon <= 1.0 && !epsilon.is_nan() else {
    raise InvalidEpsilon
  }
  let eps = @luna_cuda_pointwise_aot.canonical_cuda_float_literal(epsilon)
  let group = query_heads / kv_heads
  @utf8.encode(
    (
      $|#include <cuda_bf16.h>
      $|#include <math.h>
      $|extern "C" __global__ void \{prefix}_qknorm(__nv_bfloat16 *q, __nv_bfloat16 *k, const __nv_bfloat16 *qw, const __nv_bfloat16 *kw) {
      $|  const int row=blockIdx.x, h=blockIdx.y;
      $|  if (row>=\{rows} || h>=\{query_heads+kv_heads} || threadIdx.x!=0) return;
      $|  __nv_bfloat16 *x=h<\{query_heads} ? q : k;
      $|  const __nv_bfloat16 *w=h<\{query_heads} ? qw : kw;
      $|  const int heads=h<\{query_heads} ? \{query_heads} : \{kv_heads};
      $|  const int head=h<\{query_heads} ? h : h-\{query_heads};
      $|  const unsigned long long base=((unsigned long long)row*heads+head)*\{dimension};
      $|  float sum=0.0f;
      $|  for (int d=0;d<\{dimension};++d) { float v=__bfloat162float(x[base+d]); sum=__fadd_rn(sum,__fmul_rn(v,v)); }
      $|  const float inv=rsqrtf(__fadd_rn(__fdiv_rn(sum,\{dimension}.0f),\{eps}));
      $|  for (int d=0;d<\{dimension};++d) {
      $|    float v=__bfloat162float(__float2bfloat16_rn(__fmul_rn(__bfloat162float(x[base+d]),inv)));
      $|    x[base+d]=__float2bfloat16_rn(__fmul_rn(v,__bfloat162float(w[d])));
      $|  }
      $|}
      $|extern "C" __global__ void \{prefix}_rotary(__nv_bfloat16 *q, __nv_bfloat16 *k, const __nv_bfloat16 *cosine, const __nv_bfloat16 *sine) {
      $|  const int row=blockIdx.x,h=blockIdx.y,d=threadIdx.x;
      $|  if(row>=\{rows} || h>=\{query_heads+kv_heads} || d>=\{dimension/2}) return;
      $|  __nv_bfloat16 *x=h<\{query_heads} ? q : k;
      $|  const int heads=h<\{query_heads} ? \{query_heads} : \{kv_heads};
      $|  const int head=h<\{query_heads} ? h : h-\{query_heads};
      $|  const unsigned long long base=((unsigned long long)row*heads+head)*\{dimension};
      $|  const unsigned long long p=(unsigned long long)row*\{dimension}+d;
      $|  const float a=__bfloat162float(x[base+d]), b=__bfloat162float(x[base+d+\{dimension/2}]);
      $|  const float ac=__bfloat162float(__float2bfloat16_rn(__fmul_rn(a,__bfloat162float(cosine[p]))));
      $|  const float bs=__bfloat162float(__float2bfloat16_rn(__fmul_rn(-b,__bfloat162float(sine[p]))));
      $|  const float bc=__bfloat162float(__float2bfloat16_rn(__fmul_rn(b,__bfloat162float(cosine[p+\{dimension/2}]))));
      $|  const float as=__bfloat162float(__float2bfloat16_rn(__fmul_rn(a,__bfloat162float(sine[p+\{dimension/2}]))));
      $|  x[base+d]=__float2bfloat16_rn(__fadd_rn(ac,bs));
      $|  x[base+d+\{dimension/2}]=__float2bfloat16_rn(__fadd_rn(bc,as));
      $|}
      $|extern "C" __global__ void \{prefix}_attention(const __nv_bfloat16 *q, const __nv_bfloat16 *k, const __nv_bfloat16 *v, __nv_bfloat16 *out) {
      $|  const int row=blockIdx.x,h=blockIdx.y;
      $|  if(row>=\{rows} || h>=\{query_heads} || threadIdx.x!=0) return;
      $|  const int kh=h/\{group};
      $|  const unsigned long long qb=((unsigned long long)row*\{query_heads}+h)*\{dimension};
      $|  float accum[\{dimension}]={0}, maximum=-INFINITY, denominator=0.0f;
      $|  for(int key=0;key<=row;++key) {
      $|    const unsigned long long kb=((unsigned long long)key*\{kv_heads}+kh)*\{dimension};
      $|    float score=0.0f;
      $|    for(int d=0;d<\{dimension};++d) score=__fadd_rn(score,__fmul_rn(__bfloat162float(q[qb+d]),__bfloat162float(k[kb+d])));
      $|    score=__fmul_rn(score,rsqrtf(\{dimension}.0f));
      $|    const float next=fmaxf(maximum,score), old=expf(maximum-next), p=expf(score-next);
      $|    denominator=__fadd_rn(__fmul_rn(denominator,old),p);
      $|    for(int d=0;d<\{dimension};++d) accum[d]=__fadd_rn(__fmul_rn(accum[d],old),__fmul_rn(p,__bfloat162float(v[kb+d])));
      $|    maximum=next;
      $|  }
      $|  for(int d=0;d<\{dimension};++d) out[qb+d]=__float2bfloat16_rn(__fdiv_rn(accum[d],denominator));
      $|}
    ),
  )
}
