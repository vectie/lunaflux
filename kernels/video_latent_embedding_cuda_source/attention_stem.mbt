///|
/// Correctness-grade mixed-F32/BF16 transformer attention stem. Input and gamma
/// are F32. Normalization is F32; projection operands and result are BF16, as
/// under autocast. Q/K normalization and rotary each retain BF16 boundaries.
pub fn attention_stem_source(
  temporal~ : Int,
  height~ : Int,
  width~ : Int,
  heads~ : Int,
  head_dimension~ : Int,
  rotary_dimension~ : Int,
  register_rows~ : Int,
  prefix~ : String,
) -> Bytes raise EmbeddingError {
  guard temporal > 0 &&
    height > 0 &&
    width > 0 &&
    heads > 0 &&
    head_dimension > 0 &&
    head_dimension <= 256 &&
    rotary_dimension > 0 &&
    rotary_dimension <= head_dimension &&
    rotary_dimension % 6 == 0 &&
    temporal.to_int64() * height.to_int64() <= 2147483647L / width.to_int64() &&
    heads <= 2147483647 / head_dimension else {
    raise InvalidGeometry
  }
  let positions = temporal * height * width
  let hidden = heads * head_dimension
  let validated = render(
    batch=1,
    positions~,
    input_channels=hidden,
    hidden_channels=hidden,
    register_rows~,
    symbol=prefix + "_norm",
  )
  guard validated.output_elements <= 2147483647L / 3L else {
    raise InvalidGeometry
  }
  let rows = positions + register_rows + 1
  let elements = validated.output_elements
  let half = rotary_dimension / 2
  let axis_pairs = rotary_dimension / 6
  @utf8.encode(
    (
      $|#include <cuda_bf16.h>
      $|#include <math.h>
      $|extern "C" __global__ void \{prefix}_norm(const float *input, const float *gamma, __nv_bfloat16 *output) {
      $|  const unsigned row = blockIdx.x;
      $|  if (row >= \{rows}U || threadIdx.x != 0) return;
      $|  float sum = 0.0f;
      $|  for (int c = 0; c < \{hidden}; ++c) { const float v = input[(unsigned long long)row * \{hidden}ULL + c]; sum = __fadd_rn(sum, __fmul_rn(v,v)); }
      $|  const float scale = rsqrtf(__fadd_rn(__fdiv_rn(sum, \{hidden}.0f), 1.0e-5f));
      $|  for (int c = 0; c < \{hidden}; ++c) output[(unsigned long long)row * \{hidden}ULL + c] = __float2bfloat16_rn(__fmul_rn(__fmul_rn(input[(unsigned long long)row * \{hidden}ULL + c], scale), gamma[c]));
      $|}
      $|extern "C" __global__ void \{prefix}_qkv(const __nv_bfloat16 *input, const float *qw, const float *qb, const float *kw, const float *kb, const float *vw, const float *vb, __nv_bfloat16 *output) {
      $|  const unsigned long long flat = (unsigned long long)blockIdx.x * 256ULL + threadIdx.x;
      $|  if (flat >= \{elements * 3L}ULL) return;
      $|  const unsigned long long plane = flat / \{elements}ULL;
      $|  const unsigned long long cell = flat % \{elements}ULL;
      $|  const unsigned long long row = cell / \{hidden}ULL, c = cell % \{hidden}ULL;
      $|  const float *weight = plane == 0 ? qw : (plane == 1 ? kw : vw);
      $|  const float *bias = plane == 0 ? qb : (plane == 1 ? kb : vb);
      $|  float sum = 0.0f;
      $|  for (int k = 0; k < \{hidden}; ++k) sum = __fadd_rn(sum, __fmul_rn(__bfloat162float(input[row * \{hidden}ULL + k]), __bfloat162float(__float2bfloat16_rn(weight[c * \{hidden}ULL + k]))));
      $|  output[flat] = __float2bfloat16_rn(__fadd_rn(sum, __bfloat162float(__float2bfloat16_rn(bias[c]))));
      $|}
      $|extern "C" __global__ void \{prefix}_rotary_table(__nv_bfloat16 *table) {
      $|  const unsigned long long flat = (unsigned long long)blockIdx.x * 256ULL + threadIdx.x;
      $|  if (flat >= \{rows.to_int64() * rotary_dimension.to_int64()}ULL) return;
      $|  const unsigned long long row = flat / \{rotary_dimension}ULL;
      $|  const int slot = flat % \{rotary_dimension}ULL, pair = slot % \{half};
      $|  const int axis = pair / \{axis_pairs}, frequency = pair % \{axis_pairs};
      $|  float position = 0.0f;
      $|  if (row < \{positions}ULL) {
      $|    const int coordinate = axis == 0 ? row / (\{height}ULL * \{width}ULL) : (axis == 1 ? (row / \{width}ULL) % \{height}ULL : row % \{width}ULL);
      $|    const int extent = axis == 0 ? \{temporal} : (axis == 1 ? \{height} : \{width});
      $|    position = __fadd_rn(__fmul_rn(2.0f, __fdiv_rn(__fadd_rn((float)coordinate, 0.5f), (float)extent)), -1.0f);
      $|  }
      $|  const float inv = __fdiv_rn(1.0f, powf(100.0f, __fdiv_rn((float)frequency, \{axis_pairs}.0f)));
      $|  const float angle = __fmul_rn(__fmul_rn(6.2831853071795864769f, position), inv);
      $|  table[flat] = __float2bfloat16_rn(slot < \{half} ? cosf(angle) : sinf(angle));
      $|}
      $|extern "C" __global__ void \{prefix}_qk_rope(const __nv_bfloat16 *input, const __nv_bfloat16 *table, __nv_bfloat16 *output) {
      $|  const unsigned long long group = blockIdx.x;
      $|  if (group >= \{rows.to_int64() * heads.to_int64() * 3L}ULL || threadIdx.x != 0) return;
      $|  const unsigned long long plane = group / (\{rows}ULL * \{heads}ULL), within = group % (\{rows}ULL * \{heads}ULL), row = within / \{heads}ULL;
      $|  const unsigned long long base = group * \{head_dimension}ULL;
      $|  if (plane == 2) { for (int c=0; c<\{head_dimension}; ++c) output[base+c] = input[base+c]; return; }
      $|  float values[\{head_dimension}], sum=0.0f;
      $|  for (int c=0; c<\{head_dimension}; ++c) { values[c]=__bfloat162float(input[base+c]); sum=__fadd_rn(sum,__fmul_rn(values[c],values[c])); }
      $|  const float scale=rsqrtf(__fadd_rn(__fdiv_rn(sum,\{head_dimension}.0f),1.0e-5f));
      $|  for (int c=0; c<\{head_dimension}; ++c) values[c]=__bfloat162float(__float2bfloat16_rn(__fmul_rn(values[c],scale)));
      $|  for (int c=0; c<\{head_dimension}; ++c) {
      $|    float value=values[c];
      $|    if(c<\{rotary_dimension}) {
      $|      const int pair=c%\{half};
      $|      const float rotated=c<\{half} ? -values[c+\{half}] : values[c-\{half}];
      $|      const float a=__bfloat162float(__float2bfloat16_rn(__fmul_rn(value,__bfloat162float(table[row*\{rotary_dimension}ULL+pair]))));
      $|      const float b=__bfloat162float(__float2bfloat16_rn(__fmul_rn(rotated,__bfloat162float(table[row*\{rotary_dimension}ULL+\{half}+pair]))));
      $|      value=__fadd_rn(a,b);
      $|    }
      $|    output[base+c]=__float2bfloat16_rn(value);
      $|  }
      $|}
    ),
  )
}
