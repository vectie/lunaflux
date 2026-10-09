///|
fn valid_recurrent_delta_symbol(value : StringView) -> Bool {
  !value.is_empty() &&
  value.length() <= 128 &&
  value[0] is ('a'..='z' | 'A'..='Z' | '_') &&
  value.all(c => c is ('a'..='z' | 'A'..='Z' | '0'..='9' | '_'))
}

///|
fn recurrent_positive_product(left : Int64, right : Int64) -> Int64? {
  if left <= 0L || right <= 0L || left > 9223372036854775807L / right {
    None
  } else {
    Some(left * right)
  }
}

///|
/// Renders one deliberately serial correctness kernel. Q/K/V and output use
/// token-major `[row, head, component]` BF16 layouts. Log decay is F32
/// `[row, head, key-component]`, beta is BF16 `[row, head]`, and explicit
/// initial/final state is F32 `[sequence, head, key-component,
/// value-component]`. Sequence offsets partition the live row prefix.
pub fn causal_normalized_recurrent_delta_bf16_f32_source(
  head_count~ : Int,
  key_dimension~ : Int,
  value_dimension~ : Int,
  normalization_epsilon~ : Double,
  log_decay_lower_bound~ : Double,
  maximum_rows~ : Int,
  maximum_sequences~ : Int,
  function_symbol~ : String,
) -> RecurrentDeltaCudaSource raise RecurrentDeltaCudaSourceError {
  if head_count <= 0 {
    raise InvalidHeadCount
  }
  if key_dimension <= 0 {
    raise InvalidKeyDimension
  }
  if value_dimension <= 0 {
    raise InvalidValueDimension
  }
  if normalization_epsilon.is_nan() ||
    normalization_epsilon.is_inf() ||
    normalization_epsilon <= 0.0 {
    raise InvalidNormalizationEpsilon
  }
  if log_decay_lower_bound.is_nan() ||
    log_decay_lower_bound.is_inf() ||
    log_decay_lower_bound >= 0.0 {
    raise InvalidLogDecayLowerBound
  }
  if maximum_rows <= 0 || maximum_rows > 2147483647 {
    raise InvalidRowCount
  }
  if maximum_sequences <= 0 || maximum_sequences > maximum_rows {
    raise InvalidSequenceCount
  }
  guard recurrent_positive_product(
      head_count.to_int64(),
      key_dimension.to_int64(),
    )
    is Some(query_width) &&
    recurrent_positive_product(
      head_count.to_int64(),
      value_dimension.to_int64(),
    )
    is Some(value_width) &&
    recurrent_positive_product(
      key_dimension.to_int64(),
      value_dimension.to_int64(),
    )
    is Some(head_state_width) &&
    recurrent_positive_product(head_count.to_int64(), head_state_width)
    is Some(state_width) &&
    recurrent_positive_product(maximum_rows.to_int64(), query_width) is Some(_) &&
    recurrent_positive_product(maximum_rows.to_int64(), value_width) is Some(_) &&
    recurrent_positive_product(maximum_sequences.to_int64(), state_width)
    is Some(_) else {
    raise DimensionOverflow
  }
  if !valid_recurrent_delta_symbol(function_symbol) {
    raise InvalidFunctionSymbol
  }
  let source = @utf8.encode(
    "#include <cuda_bf16.h>\n#include <math.h>\n#include <stdint.h>\n" +
    "extern \"C\" __global__ void " +
    function_symbol +
    "(const int32_t *counts, const int32_t *sequence_offsets, const __nv_bfloat16 *query, const __nv_bfloat16 *key, const __nv_bfloat16 *value, const float *log_decay, const __nv_bfloat16 *beta, const float *initial_state, __nv_bfloat16 *output, float *final_state) {\n" +
    "  if (blockIdx.x != 0 || threadIdx.x != 0) return;\n" +
    "  const int32_t sequence_count = counts[2]; const int32_t live_rows = counts[3];\n" +
    "  if (counts[0] < 0 || counts[1] < 0 || sequence_count <= 0 || sequence_count > " +
    maximum_sequences.to_string() +
    " || counts[0] > sequence_count || counts[1] != sequence_count - counts[0] || live_rows <= 0 || live_rows > " +
    maximum_rows.to_string() +
    ") return;\n" +
    "  if (sequence_offsets[0] != 0 || sequence_offsets[sequence_count] != live_rows) return;\n" +
    "  for (int32_t s = 0; s < sequence_count; s += 1) { const int32_t begin = sequence_offsets[s]; const int32_t end = sequence_offsets[s + 1]; if (begin < 0 || end <= begin || end > live_rows) return; }\n" +
    "  const unsigned long long q_width = " +
    query_width.to_string() +
    "ULL; const unsigned long long v_width = " +
    value_width.to_string() +
    "ULL;\n" +
    "  for (int32_t row = 0; row < live_rows; row += 1) for (int32_t h = 0; h < " +
    head_count.to_string() +
    "; h += 1) {\n" +
    "    const unsigned long long qbase = (unsigned long long)row * q_width + (unsigned long long)h * " +
    key_dimension.to_string() +
    "ULL; const unsigned long long vbase = (unsigned long long)row * v_width + (unsigned long long)h * " +
    value_dimension.to_string() +
    "ULL;\n" +
    "    const float b = __bfloat162float(beta[(unsigned long long)row * " +
    head_count.to_string() +
    "ULL + h]); if (!isfinite(b) || b < 0.0f || b > 1.0f) return;\n" +
    "    for (int32_t k = 0; k < " +
    key_dimension.to_string() +
    "; k += 1) { const float q = __bfloat162float(query[qbase + k]); const float x = __bfloat162float(key[qbase + k]); const float g = log_decay[qbase + k]; if (!isfinite(q) || !isfinite(x) || !isfinite(g) || g < " +
    log_decay_lower_bound.to_string() +
    "f || g > 0.0f) return; }\n" +
    "    for (int32_t v = 0; v < " +
    value_dimension.to_string() +
    "; v += 1) if (!isfinite(__bfloat162float(value[vbase + v]))) return;\n" +
    "  }\n" +
    "  const unsigned long long live_state = (unsigned long long)sequence_count * " +
    state_width.to_string() +
    "ULL; for (unsigned long long i = 0; i < live_state; i += 1ULL) { if (!isfinite(initial_state[i])) return; }\n" +
    "  for (unsigned long long i = 0; i < live_state; i += 1ULL) final_state[i] = initial_state[i];\n" +
    "  const float q_scale = __fdiv_rn(1.0f, sqrtf(" +
    key_dimension.to_string() +
    ".0f)); const float eps = " +
    normalization_epsilon.to_string() +
    "f;\n" +
    "  for (int32_t s = 0; s < sequence_count; s += 1) for (int32_t row = sequence_offsets[s]; row < sequence_offsets[s + 1]; row += 1) for (int32_t h = 0; h < " +
    head_count.to_string() +
    "; h += 1) {\n" +
    "    const unsigned long long qbase = (unsigned long long)row * q_width + (unsigned long long)h * " +
    key_dimension.to_string() +
    "ULL; const unsigned long long vbase = (unsigned long long)row * v_width + (unsigned long long)h * " +
    value_dimension.to_string() +
    "ULL; const unsigned long long state = ((unsigned long long)s * " +
    head_count.to_string() +
    "ULL + h) * " +
    head_state_width.to_string() +
    "ULL;\n" +
    "    float qsq = 0.0f; float ksq = 0.0f; for (int32_t k = 0; k < " +
    key_dimension.to_string() +
    "; k += 1) { const float q = __bfloat162float(query[qbase + k]); const float x = __bfloat162float(key[qbase + k]); qsq = __fadd_rn(qsq, __fmul_rn(q, q)); ksq = __fadd_rn(ksq, __fmul_rn(x, x)); }\n" +
    "    const float qinv = __fdiv_rn(1.0f, sqrtf(__fadd_rn(qsq, eps))); const float kinv = __fdiv_rn(1.0f, sqrtf(__fadd_rn(ksq, eps)));\n" +
    "    for (int32_t k = 0; k < " +
    key_dimension.to_string() +
    "; k += 1) { const float decay = expf(log_decay[qbase + k]); for (int32_t v = 0; v < " +
    value_dimension.to_string() +
    "; v += 1) { const unsigned long long i = state + (unsigned long long)k * " +
    value_dimension.to_string() +
    "ULL + v; final_state[i] = __fmul_rn(final_state[i], decay); } }\n" +
    "    const float b = __bfloat162float(beta[(unsigned long long)row * " +
    head_count.to_string() +
    "ULL + h]);\n" +
    "    for (int32_t v = 0; v < " +
    value_dimension.to_string() +
    "; v += 1) { float memory = 0.0f; for (int32_t k = 0; k < " +
    key_dimension.to_string() +
    "; k += 1) { const float kn = __fmul_rn(__bfloat162float(key[qbase + k]), kinv); memory = __fadd_rn(memory, __fmul_rn(final_state[state + (unsigned long long)k * " +
    value_dimension.to_string() +
    "ULL + v], kn)); } const float delta = __fmul_rn(__fsub_rn(__bfloat162float(value[vbase + v]), memory), b); for (int32_t k = 0; k < " +
    key_dimension.to_string() +
    "; k += 1) { const unsigned long long i = state + (unsigned long long)k * " +
    value_dimension.to_string() +
    "ULL + v; const float kn = __fmul_rn(__bfloat162float(key[qbase + k]), kinv); final_state[i] = __fadd_rn(final_state[i], __fmul_rn(kn, delta)); } }\n" +
    "    for (int32_t v = 0; v < " +
    value_dimension.to_string() +
    "; v += 1) { float projected = 0.0f; for (int32_t k = 0; k < " +
    key_dimension.to_string() +
    "; k += 1) { const float qn = __fmul_rn(__fmul_rn(__bfloat162float(query[qbase + k]), qinv), q_scale); projected = __fadd_rn(projected, __fmul_rn(final_state[state + (unsigned long long)k * " +
    value_dimension.to_string() +
    "ULL + v], qn)); } output[vbase + v] = __float2bfloat16_rn(projected); }\n" +
    "  }\n}\n",
  )
  {
    operation: CausalNormalizedRecurrentDeltaBf16F32,
    numerical_contract: OrderedFloat32NormalizedDeltaBf16OutputV1,
    head_count,
    key_dimension,
    value_dimension,
    normalization_epsilon,
    log_decay_lower_bound,
    maximum_rows,
    maximum_sequences,
    function_symbol,
    source,
  }
}

///|
pub fn RecurrentDeltaCudaSource::operation(
  self : Self,
) -> RecurrentDeltaOperation {
  self.operation
}

///|
pub fn RecurrentDeltaCudaSource::numerical_contract(
  self : Self,
) -> RecurrentDeltaNumericalContract {
  self.numerical_contract
}

///|
pub fn RecurrentDeltaCudaSource::head_count(self : Self) -> Int {
  self.head_count
}

///|
pub fn RecurrentDeltaCudaSource::key_dimension(self : Self) -> Int {
  self.key_dimension
}

///|
pub fn RecurrentDeltaCudaSource::value_dimension(self : Self) -> Int {
  self.value_dimension
}

///|
pub fn RecurrentDeltaCudaSource::normalization_epsilon(self : Self) -> Double {
  self.normalization_epsilon
}

///|
pub fn RecurrentDeltaCudaSource::log_decay_lower_bound(self : Self) -> Double {
  self.log_decay_lower_bound
}

///|
pub fn RecurrentDeltaCudaSource::maximum_rows(self : Self) -> Int {
  self.maximum_rows
}

///|
pub fn RecurrentDeltaCudaSource::maximum_sequences(self : Self) -> Int {
  self.maximum_sequences
}

///|
pub fn RecurrentDeltaCudaSource::function_symbol(self : Self) -> String {
  self.function_symbol
}

///|
pub fn RecurrentDeltaCudaSource::source_bytes(self : Self) -> Bytes {
  self.source
}
