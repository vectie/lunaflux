///|
fn valid_recurrent_projection_symbol(value : StringView) -> Bool {
  !value.is_empty() &&
  value.length() <= 128 &&
  value[0] is ('a'..='z' | 'A'..='Z' | '_') &&
  value.all(c => c is ('a'..='z' | 'A'..='Z' | '0'..='9' | '_'))
}

///|
fn recurrent_projection_product(left : Int64, right : Int64) -> Int64? {
  if left <= 0L || right <= 0L || left > 9223372036854775807L / right {
    None
  } else {
    Some(left * right)
  }
}

///|
/// Renders a deliberately serial correctness kernel. Hidden input is BF16
/// `[maximum_rows, hidden_width]`. Q/K/V weights are independent output-major
/// BF16 `[head_count*head_dimension, hidden_width]`; forget-A is
/// `[control_rank, hidden_width]`, forget-B is
/// `[head_count*head_dimension, control_rank]`, and input-gate weight is
/// `[head_count, hidden_width]`. Outputs are token-major BF16 Q/K/V, explicit
/// forget-A, raw forget logits, and sigmoid input gate. F32 decay parameters
/// and log-decay production are intentionally absent.
pub fn qkv_forget_input_gate_bf16_source(
  hidden_width~ : Int,
  head_count~ : Int,
  head_dimension~ : Int,
  control_rank~ : Int,
  maximum_rows~ : Int,
  function_symbol~ : String,
) -> RecurrentAttentionProjectionCudaSource raise RecurrentAttentionProjectionCudaSourceError {
  if hidden_width <= 0 {
    raise InvalidHiddenWidth
  }
  if head_count <= 0 {
    raise InvalidHeadCount
  }
  if head_dimension <= 0 {
    raise InvalidHeadDimension
  }
  if control_rank <= 0 {
    raise InvalidControlRank
  }
  if maximum_rows <= 0 || maximum_rows > 2147483647 {
    raise InvalidRowCount
  }
  guard recurrent_projection_product(
      head_count.to_int64(),
      head_dimension.to_int64(),
    )
    is Some(projected_width) &&
    recurrent_projection_product(
      maximum_rows.to_int64(),
      hidden_width.to_int64(),
    )
    is Some(_) &&
    recurrent_projection_product(maximum_rows.to_int64(), projected_width)
    is Some(_) &&
    recurrent_projection_product(
      maximum_rows.to_int64(),
      control_rank.to_int64(),
    )
    is Some(_) &&
    recurrent_projection_product(maximum_rows.to_int64(), head_count.to_int64())
    is Some(_) &&
    recurrent_projection_product(projected_width, hidden_width.to_int64())
    is Some(_) &&
    recurrent_projection_product(
      control_rank.to_int64(),
      hidden_width.to_int64(),
    )
    is Some(_) &&
    recurrent_projection_product(projected_width, control_rank.to_int64())
    is Some(_) &&
    recurrent_projection_product(head_count.to_int64(), hidden_width.to_int64())
    is Some(_) else {
    raise DimensionOverflow
  }
  if !valid_recurrent_projection_symbol(function_symbol) {
    raise InvalidFunctionSymbol
  }
  let source = @utf8.encode(
    "#include <cuda_bf16.h>\n#include <math.h>\n#include <stdint.h>\n" +
    "extern \"C\" __global__ void " +
    function_symbol +
    "(const int32_t *counts, const __nv_bfloat16 *hidden, const __nv_bfloat16 *q_weight, const __nv_bfloat16 *k_weight, const __nv_bfloat16 *v_weight, const __nv_bfloat16 *forget_a_weight, const __nv_bfloat16 *forget_b_weight, const __nv_bfloat16 *input_gate_weight, __nv_bfloat16 *query, __nv_bfloat16 *key, __nv_bfloat16 *value, __nv_bfloat16 *forget_low_rank, __nv_bfloat16 *raw_forget, __nv_bfloat16 *beta) {\n" +
    "  if (blockIdx.x != 0 || threadIdx.x != 0) return; const int32_t live_rows = counts[3];\n" +
    "  if (counts[0] < 0 || counts[1] < 0 || counts[2] <= 0 || counts[0] > counts[2] || counts[1] != counts[2] - counts[0] || live_rows <= 0 || live_rows > " +
    maximum_rows.to_string() +
    ") return;\n" +
    "  for (int32_t row = 0; row < live_rows; row += 1) { const unsigned long long hidden_base = (unsigned long long)row * " +
    hidden_width.to_string() +
    "ULL; const unsigned long long projected_base = (unsigned long long)row * " +
    projected_width.to_string() +
    "ULL;\n" +
    "    for (int32_t out = 0; out < " +
    projected_width.to_string() +
    "; out += 1) { float qsum = 0.0f; float ksum = 0.0f; float vsum = 0.0f; const unsigned long long weight_base = (unsigned long long)out * " +
    hidden_width.to_string() +
    "ULL; for (int32_t in = 0; in < " +
    hidden_width.to_string() +
    "; in += 1) { const float x = __bfloat162float(hidden[hidden_base + in]); qsum = __fadd_rn(qsum, __fmul_rn(x, __bfloat162float(q_weight[weight_base + in]))); ksum = __fadd_rn(ksum, __fmul_rn(x, __bfloat162float(k_weight[weight_base + in]))); vsum = __fadd_rn(vsum, __fmul_rn(x, __bfloat162float(v_weight[weight_base + in]))); } query[projected_base + out] = __float2bfloat16_rn(qsum); key[projected_base + out] = __float2bfloat16_rn(ksum); value[projected_base + out] = __float2bfloat16_rn(vsum); }\n" +
    "    const unsigned long long low_base = (unsigned long long)row * " +
    control_rank.to_string() +
    "ULL; for (int32_t out = 0; out < " +
    control_rank.to_string() +
    "; out += 1) { float sum = 0.0f; const unsigned long long weight_base = (unsigned long long)out * " +
    hidden_width.to_string() +
    "ULL; for (int32_t in = 0; in < " +
    hidden_width.to_string() +
    "; in += 1) sum = __fadd_rn(sum, __fmul_rn(__bfloat162float(hidden[hidden_base + in]), __bfloat162float(forget_a_weight[weight_base + in]))); forget_low_rank[low_base + out] = __float2bfloat16_rn(sum); }\n" +
    "    for (int32_t out = 0; out < " +
    projected_width.to_string() +
    "; out += 1) { float sum = 0.0f; const unsigned long long weight_base = (unsigned long long)out * " +
    control_rank.to_string() +
    "ULL; for (int32_t in = 0; in < " +
    control_rank.to_string() +
    "; in += 1) sum = __fadd_rn(sum, __fmul_rn(__bfloat162float(forget_low_rank[low_base + in]), __bfloat162float(forget_b_weight[weight_base + in]))); raw_forget[projected_base + out] = __float2bfloat16_rn(sum); }\n" +
    "    const unsigned long long gate_base = (unsigned long long)row * " +
    head_count.to_string() +
    "ULL; for (int32_t out = 0; out < " +
    head_count.to_string() +
    "; out += 1) { float sum = 0.0f; const unsigned long long weight_base = (unsigned long long)out * " +
    hidden_width.to_string() +
    "ULL; for (int32_t in = 0; in < " +
    hidden_width.to_string() +
    "; in += 1) sum = __fadd_rn(sum, __fmul_rn(__bfloat162float(hidden[hidden_base + in]), __bfloat162float(input_gate_weight[weight_base + in]))); const __nv_bfloat16 logit = __float2bfloat16_rn(sum); const float sigmoid = __fdiv_rn(1.0f, __fadd_rn(1.0f, expf(-__bfloat162float(logit)))); beta[gate_base + out] = __float2bfloat16_rn(sigmoid); }\n" +
    "  }\n}\n",
  )
  {
    operation: QkvForgetAndInputGateBf16,
    numerical_contract: OrderedFloat32DenseBf16IntermediateSigmoidV1,
    hidden_width,
    head_count,
    head_dimension,
    control_rank,
    maximum_rows,
    function_symbol,
    source,
  }
}

///|
pub fn RecurrentAttentionProjectionCudaSource::operation(
  self : Self,
) -> RecurrentAttentionProjectionOperation {
  self.operation
}

///|
pub fn RecurrentAttentionProjectionCudaSource::numerical_contract(
  self : Self,
) -> RecurrentAttentionProjectionNumericalContract {
  self.numerical_contract
}

///|
pub fn RecurrentAttentionProjectionCudaSource::hidden_width(self : Self) -> Int {
  self.hidden_width
}

///|
pub fn RecurrentAttentionProjectionCudaSource::head_count(self : Self) -> Int {
  self.head_count
}

///|
pub fn RecurrentAttentionProjectionCudaSource::head_dimension(
  self : Self,
) -> Int {
  self.head_dimension
}

///|
pub fn RecurrentAttentionProjectionCudaSource::control_rank(self : Self) -> Int {
  self.control_rank
}

///|
pub fn RecurrentAttentionProjectionCudaSource::maximum_rows(self : Self) -> Int {
  self.maximum_rows
}

///|
pub fn RecurrentAttentionProjectionCudaSource::function_symbol(
  self : Self,
) -> String {
  self.function_symbol
}

///|
pub fn RecurrentAttentionProjectionCudaSource::source_bytes(
  self : Self,
) -> Bytes {
  self.source
}
