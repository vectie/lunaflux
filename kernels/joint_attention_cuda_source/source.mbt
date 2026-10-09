///|
fn valid_symbol(value : StringView) -> Bool {
  !value.is_empty() &&
  value.length() <= 96 &&
  value[0] is ('a'..='z' | 'A'..='Z' | '_') &&
  value.all(c => c is ('a'..='z' | 'A'..='Z' | '0'..='9' | '_'))
}

///|
/// Renders four staged BF16 attention entrypoints. Q/K/V and output weights
/// are row-major and bias-free. Q/K RMSNorm is per head. Attention is full,
/// noncausal within each document. Optional cumulative row boundaries isolate
/// packed documents without a dense mask or a device-side metadata upload.
pub fn staged_joint_attention_bf16_source(
  rows~ : Int,
  hidden_size~ : Int,
  head_count~ : Int,
  head_dimension~ : Int,
  rotary_dimension~ : Int,
  epsilon_numerator~ : Int,
  epsilon_denominator~ : Int,
  apply_rotary~ : Bool,
  function_prefix~ : String,
  document_boundaries? : ArrayView[Int],
) -> JointAttentionCudaSource raise JointAttentionCudaSourceError {
  if rows <= 0 {
    raise InvalidRowCount
  }
  let (document_scope, key_start, key_end) = match document_boundaries {
    None => ("", "0", rows.to_string())
    Some(boundaries) => {
      guard boundaries.length() >= 2 &&
        boundaries[0] == 0 &&
        boundaries[boundaries.length() - 1] == rows else {
        raise InvalidDocumentBoundaries
      }
      for i in 1..<boundaries.length() {
        guard boundaries[i] > boundaries[i - 1] else {
          raise InvalidDocumentBoundaries
        }
      }
      if boundaries.length() == 2 {
        ("", "0", rows.to_string())
      } else {
        let clauses : Array[String] = []
        for i in 1..<(boundaries.length() - 1) {
          clauses.push(
            "if (qr >= " +
            boundaries[i].to_string() +
            "ULL) { key_begin = " +
            boundaries[i].to_string() +
            "; key_end = " +
            boundaries[i + 1].to_string() +
            "; }",
          )
        }
        (
          " int key_begin = 0; int key_end = " +
          boundaries[1].to_string() +
          "; " +
          clauses.join(" "),
          "key_begin",
          "key_end",
        )
      }
    }
  }
  if hidden_size <= 0 ||
    head_count <= 0 ||
    head_dimension <= 0 ||
    rotary_dimension < 0 ||
    rotary_dimension > head_dimension ||
    rotary_dimension % 2 != 0 ||
    (apply_rotary && rotary_dimension == 0) ||
    (!apply_rotary && rotary_dimension != 0) {
    raise InvalidWidth
  }
  if epsilon_numerator <= 0 || epsilon_denominator <= 0 {
    raise InvalidEpsilon
  }
  if head_count.to_int64() > 2147483647L / head_dimension.to_int64() ||
    rows.to_int64() > 2147483647L / hidden_size.to_int64() {
    raise DimensionOverflow
  }
  if !valid_symbol(function_prefix) {
    raise InvalidFunctionSymbol
  }
  let inner = head_count * head_dimension
  let hidden_elements = rows.to_int64() * hidden_size.to_int64()
  let inner_elements = rows.to_int64() * inner.to_int64()
  let row_heads = rows.to_int64() * head_count.to_int64()
  let epsilon = epsilon_numerator.to_double() / epsilon_denominator.to_double()
  let scale = 1.0 / head_dimension.to_double().sqrt()
  let norm_signature = if apply_rotary {
    "(const __nv_bfloat16 *query, const __nv_bfloat16 *key, const __nv_bfloat16 *query_weight, const __nv_bfloat16 *key_weight, const float *rotary_cos, const float *rotary_sin, __nv_bfloat16 *normalized_query, __nv_bfloat16 *normalized_key)"
  } else {
    "(const __nv_bfloat16 *query, const __nv_bfloat16 *key, const __nv_bfloat16 *query_weight, const __nv_bfloat16 *key_weight, __nv_bfloat16 *normalized_query, __nv_bfloat16 *normalized_key)"
  }
  let rotary_body = if apply_rotary {
    " if (column < " +
    rotary_dimension.to_string() +
    ") { const int half = " +
    (rotary_dimension / 2).to_string() +
    "; const int pair = column < half ? column + half : column - half; const float sign = column < half ? -1.0f : 1.0f; const __nv_bfloat16 cosine = __float2bfloat16_rn(rotary_cos[row * " +
    rotary_dimension.to_string() +
    "ULL + column]); const __nv_bfloat16 sine = __float2bfloat16_rn(rotary_sin[row * " +
    rotary_dimension.to_string() +
    "ULL + column]); const __nv_bfloat16 qr = __float2bfloat16_rn(__fmul_rn(sign, __bfloat162float(q_values[pair]))); const __nv_bfloat16 kr = __float2bfloat16_rn(__fmul_rn(sign, __bfloat162float(k_values[pair]))); q_value = __float2bfloat16_rn(__fadd_rn(__bfloat162float(__float2bfloat16_rn(__fmul_rn(__bfloat162float(q_values[column]), __bfloat162float(cosine)))), __bfloat162float(__float2bfloat16_rn(__fmul_rn(__bfloat162float(qr), __bfloat162float(sine)))))); k_value = __float2bfloat16_rn(__fadd_rn(__bfloat162float(__float2bfloat16_rn(__fmul_rn(__bfloat162float(k_values[column]), __bfloat162float(cosine)))), __bfloat162float(__float2bfloat16_rn(__fmul_rn(__bfloat162float(kr), __bfloat162float(sine)))))); }"
  } else {
    ""
  }
  let source = @utf8.encode(
    "#include <cuda_bf16.h>\n#include <math.h>\n#include <stdint.h>\n" +
    "extern \"C\" __global__ void " +
    function_prefix +
    "_qkv(const __nv_bfloat16 *input, const __nv_bfloat16 *query_weight, const __nv_bfloat16 *key_weight, const __nv_bfloat16 *value_weight, __nv_bfloat16 *query, __nv_bfloat16 *key, __nv_bfloat16 *value) { const unsigned long long i = (unsigned long long)blockIdx.x * 256ULL + threadIdx.x; if (i >= " +
    inner_elements.to_string() +
    "ULL) return; const unsigned long long row = i / " +
    inner.to_string() +
    "ULL; const int output = (int)(i % " +
    inner.to_string() +
    "ULL); float q = 0.0f; float k = 0.0f; float v = 0.0f; const unsigned long long wb = (unsigned long long)output * " +
    hidden_size.to_string() +
    "ULL; const unsigned long long ib = row * " +
    hidden_size.to_string() +
    "ULL; for (int c = 0; c < " +
    hidden_size.to_string() +
    "; c += 1) { const float x = __bfloat162float(input[ib + c]); q = __fadd_rn(q, __fmul_rn(x, __bfloat162float(query_weight[wb + c]))); k = __fadd_rn(k, __fmul_rn(x, __bfloat162float(key_weight[wb + c]))); v = __fadd_rn(v, __fmul_rn(x, __bfloat162float(value_weight[wb + c]))); } query[i] = __float2bfloat16_rn(q); key[i] = __float2bfloat16_rn(k); value[i] = __float2bfloat16_rn(v); }\n" +
    "extern \"C\" __global__ void " +
    function_prefix +
    "_qk_norm" +
    norm_signature +
    " { const unsigned long long rh = (unsigned long long)blockIdx.x; if (rh >= " +
    row_heads.to_string() +
    "ULL || threadIdx.x != 0) return; const unsigned long long row = rh / " +
    head_count.to_string() +
    "ULL; const unsigned long long base = rh * " +
    head_dimension.to_string() +
    "ULL; float qs = 0.0f; float ks = 0.0f; for (int column = 0; column < " +
    head_dimension.to_string() +
    "; column += 1) { const float q = __bfloat162float(query[base + column]); const float k = __bfloat162float(key[base + column]); qs = __fadd_rn(qs, __fmul_rn(q, q)); ks = __fadd_rn(ks, __fmul_rn(k, k)); } const float qi = rsqrtf(__fadd_rn(__fdiv_rn(qs, " +
    head_dimension.to_string() +
    ".0f), " +
    epsilon.to_string() +
    "f)); const float ki = rsqrtf(__fadd_rn(__fdiv_rn(ks, " +
    head_dimension.to_string() +
    ".0f), " +
    epsilon.to_string() +
    "f)); __nv_bfloat16 q_values[" +
    head_dimension.to_string() +
    "]; __nv_bfloat16 k_values[" +
    head_dimension.to_string() +
    "]; for (int column = 0; column < " +
    head_dimension.to_string() +
    "; column += 1) { q_values[column] = __float2bfloat16_rn(__fmul_rn(__fmul_rn(__bfloat162float(query[base + column]), qi), __bfloat162float(query_weight[column]))); k_values[column] = __float2bfloat16_rn(__fmul_rn(__fmul_rn(__bfloat162float(key[base + column]), ki), __bfloat162float(key_weight[column]))); } for (int column = 0; column < " +
    head_dimension.to_string() +
    "; column += 1) { __nv_bfloat16 q_value = q_values[column]; __nv_bfloat16 k_value = k_values[column];" +
    rotary_body +
    " normalized_query[base + column] = q_value; normalized_key[base + column] = k_value; } }\n" +
    "extern \"C\" __global__ void " +
    function_prefix +
    "_attention(const __nv_bfloat16 *query, const __nv_bfloat16 *key, const __nv_bfloat16 *value, __nv_bfloat16 *attention_output) { const unsigned long long i = (unsigned long long)blockIdx.x * 256ULL + threadIdx.x; if (i >= " +
    inner_elements.to_string() +
    "ULL) return; const unsigned long long qr = i / " +
    inner.to_string() +
    "ULL; const int ic = (int)(i % " +
    inner.to_string() +
    "ULL); const int head = ic / " +
    head_dimension.to_string() +
    "; const int vc = ic % " +
    head_dimension.to_string() +
    "; const unsigned long long qb = qr * " +
    inner.to_string() +
    "ULL + (unsigned long long)head * " +
    head_dimension.to_string() +
    "ULL;" +
    document_scope +
    " float maximum = -INFINITY; for (int kr = " +
    key_start +
    "; kr < " +
    key_end +
    "; kr += 1) { float dot = 0.0f; const unsigned long long kb = (unsigned long long)kr * " +
    inner.to_string() +
    "ULL + (unsigned long long)head * " +
    head_dimension.to_string() +
    "ULL; for (int c = 0; c < " +
    head_dimension.to_string() +
    "; c += 1) dot = __fadd_rn(dot, __fmul_rn(__bfloat162float(query[qb + c]), __bfloat162float(key[kb + c]))); maximum = fmaxf(maximum, __fmul_rn(dot, " +
    scale.to_string() +
    "f)); } float denominator = 0.0f; float numerator = 0.0f; for (int kr = " +
    key_start +
    "; kr < " +
    key_end +
    "; kr += 1) { float dot = 0.0f; const unsigned long long kb = (unsigned long long)kr * " +
    inner.to_string() +
    "ULL + (unsigned long long)head * " +
    head_dimension.to_string() +
    "ULL; for (int c = 0; c < " +
    head_dimension.to_string() +
    "; c += 1) dot = __fadd_rn(dot, __fmul_rn(__bfloat162float(query[qb + c]), __bfloat162float(key[kb + c]))); const float probability = expf(__fsub_rn(__fmul_rn(dot, " +
    scale.to_string() +
    "f), maximum)); denominator = __fadd_rn(denominator, probability); const unsigned long long vi = (unsigned long long)kr * " +
    inner.to_string() +
    "ULL + (unsigned long long)head * " +
    head_dimension.to_string() +
    "ULL + vc; numerator = __fadd_rn(numerator, __fmul_rn(probability, __bfloat162float(value[vi]))); } attention_output[i] = __float2bfloat16_rn(__fdiv_rn(numerator, denominator)); }\n" +
    "extern \"C\" __global__ void " +
    function_prefix +
    "_output(const __nv_bfloat16 *attention_input, const __nv_bfloat16 *output_weight, __nv_bfloat16 *output) { const unsigned long long i = (unsigned long long)blockIdx.x * 256ULL + threadIdx.x; if (i >= " +
    hidden_elements.to_string() +
    "ULL) return; const unsigned long long row = i / " +
    hidden_size.to_string() +
    "ULL; const int out = (int)(i % " +
    hidden_size.to_string() +
    "ULL); float sum = 0.0f; const unsigned long long ib = row * " +
    inner.to_string() +
    "ULL; const unsigned long long wb = (unsigned long long)out * " +
    inner.to_string() +
    "ULL; for (int c = 0; c < " +
    inner.to_string() +
    "; c += 1) sum = __fadd_rn(sum, __fmul_rn(__bfloat162float(attention_input[ib + c]), __bfloat162float(output_weight[wb + c]))); output[i] = __float2bfloat16_rn(sum); }\n",
  )
  {
    numerical_contract: OrderedF32AccumulationBf16StagedAttentionV1,
    row_count: rows,
    hidden_size,
    head_count,
    head_dimension,
    rotary_dimension,
    applies_rotary: apply_rotary,
    function_prefix,
    source,
  }
}
