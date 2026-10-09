///|
fn valid_symbol(symbol : StringView) -> Bool {
  if symbol.is_empty() || symbol.length() > 128 {
    return false
  }
  symbol[0] is ('a'..='z' | 'A'..='Z' | '_') &&
  symbol.all(c => c is ('a'..='z' | 'A'..='Z' | '0'..='9' | '_'))
}

///|
fn prelude() -> String {
  "#include <cuda_bf16.h>\n#include <math.h>\n#include <stdint.h>\n"
}

///|
pub fn token_embedding_bf16_source(
  vocabulary_size~ : Int,
  hidden_size~ : Int,
  function_symbol~ : String,
) -> DecoderFoundationCudaSource raise DecoderFoundationCudaSourceError {
  if vocabulary_size <= 0 {
    raise InvalidVocabularySize
  }
  if hidden_size <= 0 {
    raise InvalidHiddenSize
  }
  if !valid_symbol(function_symbol) {
    raise InvalidFunctionSymbol
  }
  let source = @utf8.encode(
    prelude() +
    "extern \"C\" __global__ void " +
    function_symbol +
    "(const int32_t *counts, const int32_t *token_ids, const __nv_bfloat16 *table, __nv_bfloat16 *output) {\n" +
    "  const unsigned long long i = (unsigned long long)blockIdx.x * 256ULL + threadIdx.x;\n" +
    "  const int live_tokens = counts[3];\n  if (live_tokens <= 0) return;\n" +
    "  const unsigned long long live = (unsigned long long)live_tokens * " +
    hidden_size.to_string() +
    "ULL;\n  if (i >= live) return;\n" +
    "  const int32_t token = token_ids[i / " +
    hidden_size.to_string() +
    "ULL];\n" +
    "  if (token >= 0 && token < " +
    vocabulary_size.to_string() +
    ") output[i] = table[(unsigned long long)token * " +
    hidden_size.to_string() +
    "ULL + i % " +
    hidden_size.to_string() +
    "ULL];\n" +
    "  else output[i] = __float2bfloat16_rn(0.0f);\n}\n",
  )
  {
    operation: TokenEmbeddingBf16,
    numerical_contract: BitExactBf16Copy,
    function_symbol,
    source,
  }
}

///|
pub fn rms_norm_bf16_source(
  hidden_size~ : Int,
  epsilon~ : Double,
  function_symbol~ : String,
) -> DecoderFoundationCudaSource raise DecoderFoundationCudaSourceError {
  if hidden_size <= 0 {
    raise InvalidHiddenSize
  }
  if epsilon.is_nan() || epsilon.is_inf() || epsilon <= 0.0 || epsilon > 1.0 {
    raise InvalidEpsilon
  }
  if !valid_symbol(function_symbol) {
    raise InvalidFunctionSymbol
  }
  let source = @utf8.encode(
    prelude() +
    "extern \"C\" __global__ void " +
    function_symbol +
    "(const int32_t *counts, const __nv_bfloat16 *input, const __nv_bfloat16 *weight, __nv_bfloat16 *output) {\n" +
    "  const int row = (int)blockIdx.x;\n  if (counts[3] <= 0 || row >= counts[3]) return;\n" +
    "  __shared__ float partial[256];\n  float sum = 0.0f;\n" +
    "  const unsigned long long base = (unsigned long long)row * " +
    hidden_size.to_string() +
    "ULL;\n" +
    "  for (int column = threadIdx.x; column < " +
    hidden_size.to_string() +
    "; column += 256) { const float value = __bfloat162float(input[base + column]); sum = __fadd_rn(sum, __fmul_rn(value, value)); }\n" +
    "  partial[threadIdx.x] = sum;\n  __syncthreads();\n" +
    "  for (int stride = 128; stride > 0; stride >>= 1) { if (threadIdx.x < stride) partial[threadIdx.x] = __fadd_rn(partial[threadIdx.x], partial[threadIdx.x + stride]); __syncthreads(); }\n" +
    "  const float inverse = rsqrtf(__fadd_rn(__fdiv_rn(partial[0], " +
    hidden_size.to_string() +
    ".0f), " +
    @luna_cuda_pointwise_aot.canonical_cuda_float_literal(epsilon) +
    "));\n" +
    "  for (int column = threadIdx.x; column < " +
    hidden_size.to_string() +
    "; column += 256) { const float value = __bfloat162float(input[base + column]); const float scale = __bfloat162float(weight[column]); output[base + column] = __float2bfloat16_rn(__fmul_rn(__fmul_rn(value, inverse), scale)); }\n}\n",
  )
  {
    operation: RmsNormBf16,
    numerical_contract: OrderedFloat32ReductionBf16RoundV1,
    function_symbol,
    source,
  }
}
