///|
fn valid_dsa_pooling_symbol(symbol : StringView) -> Bool {
  if symbol.is_empty() || symbol.length() > 128 {
    return false
  }
  symbol[0] is ('a'..='z' | 'A'..='Z' | '_') &&
  symbol.all(c => c is ('a'..='z' | 'A'..='Z' | '0'..='9' | '_'))
}

///|
/// Renders a correctness-grade single-sequence DSA key-pool compressor. The
/// serial launch makes pool compaction and floating-point reduction order
/// explicit. Unused output capacity is deterministically cleared.
pub fn dsa_key_pool_compression_bf16_source(
  head_dimension~ : Int,
  pool_size~ : Int,
  maximum_tokens~ : Int,
  function_symbol~ : String,
) -> GlmHybridDsaPoolingCudaSource raise GlmHybridDsaPoolingCudaSourceError {
  if head_dimension <= 0 || head_dimension > 65536 {
    raise InvalidHeadDimension
  }
  if pool_size <= 0 || pool_size > 1024 {
    raise InvalidPoolSize
  }
  if maximum_tokens < pool_size || maximum_tokens > 1048576 {
    raise InvalidMaximumTokens
  }
  if !valid_dsa_pooling_symbol(function_symbol) {
    raise InvalidFunctionSymbol
  }
  let pool_capacity = maximum_tokens / pool_size
  let source = @utf8.encode(
    "#include <cuda_bf16.h>\n#include <stdint.h>\n#include <math.h>\n" +
    "extern \"C\" __global__ void " +
    function_symbol +
    "(const int32_t *counts, const __nv_bfloat16 *keys, const __nv_bfloat16 *gate_scores, const int32_t *valid_tokens, const __nv_bfloat16 *pool_ape, __nv_bfloat16 *pooled_keys, int32_t *raw_indices, int32_t *pool_valid, int32_t *pool_count) {\n" +
    "  if (blockIdx.x != 0 || threadIdx.x != 0) return;\n" +
    "  const int live_tokens = counts[3];\n" +
    "  const int head_dimension = " +
    head_dimension.to_string() +
    ";\n  const int pool_size = " +
    pool_size.to_string() +
    ";\n  const int pool_capacity = " +
    pool_capacity.to_string() +
    ";\n" +
    "  *pool_count = 0;\n" +
    "  for (int pool = 0; pool < pool_capacity; ++pool) {\n" +
    "    pool_valid[pool] = 0;\n" +
    "    for (int offset = 0; offset < pool_size; ++offset) raw_indices[pool * pool_size + offset] = -1;\n" +
    "    for (int component = 0; component < head_dimension; ++component) pooled_keys[pool * head_dimension + component] = __float2bfloat16_rn(0.0f);\n" +
    "  }\n" +
    "  if (live_tokens <= 0 || live_tokens > " +
    maximum_tokens.to_string() +
    ") return;\n" +
    "  int first_valid = live_tokens;\n" +
    "  for (int token = 0; token < live_tokens; ++token) { if (valid_tokens[token] != 0) { first_valid = token; break; } }\n" +
    "  const int complete_pools = (live_tokens - first_valid) / pool_size;\n" +
    "  int emitted = 0;\n" +
    "  for (int pool = 0; pool < complete_pools && emitted < pool_capacity; ++pool) {\n" +
    "    const int raw_start = first_valid + pool * pool_size;\n" +
    "    int all_valid = 1;\n" +
    "    for (int offset = 0; offset < pool_size; ++offset) all_valid &= valid_tokens[raw_start + offset] != 0;\n" +
    "    if (!all_valid) continue;\n" +
    "    for (int offset = 0; offset < pool_size; ++offset) raw_indices[emitted * pool_size + offset] = raw_start + offset;\n" +
    "    for (int component = 0; component < head_dimension; ++component) {\n" +
    "      float maximum = -INFINITY;\n" +
    "      for (int offset = 0; offset < pool_size; ++offset) {\n" +
    "        const int raw = raw_start + offset;\n" +
    "        const float logit = __fadd_rn(__bfloat162float(gate_scores[raw * head_dimension + component]), __bfloat162float(pool_ape[offset * head_dimension + component]));\n" +
    "        if (logit > maximum) maximum = logit;\n" +
    "      }\n" +
    "      float denominator = 0.0f;\n" +
    "      for (int offset = 0; offset < pool_size; ++offset) {\n" +
    "        const int raw = raw_start + offset;\n" +
    "        const float logit = __fadd_rn(__bfloat162float(gate_scores[raw * head_dimension + component]), __bfloat162float(pool_ape[offset * head_dimension + component]));\n" +
    "        denominator = __fadd_rn(denominator, expf(__fsub_rn(logit, maximum)));\n" +
    "      }\n" +
    "      float accumulator = 0.0f;\n" +
    "      for (int offset = 0; offset < pool_size; ++offset) {\n" +
    "        const int raw = raw_start + offset;\n" +
    "        const float logit = __fadd_rn(__bfloat162float(gate_scores[raw * head_dimension + component]), __bfloat162float(pool_ape[offset * head_dimension + component]));\n" +
    "        const __nv_bfloat16 probability = __float2bfloat16_rn(__fdividef(expf(__fsub_rn(logit, maximum)), denominator));\n" +
    "        const float product = __fmul_rn(__bfloat162float(probability), __bfloat162float(keys[raw * head_dimension + component]));\n" +
    "        accumulator = __fadd_rn(accumulator, product);\n" +
    "      }\n" +
    "      pooled_keys[emitted * head_dimension + component] = __float2bfloat16_rn(accumulator);\n" +
    "    }\n" +
    "    pool_valid[emitted] = 1;\n    ++emitted;\n" +
    "  }\n  *pool_count = emitted;\n}\n",
  )
  {
    operation: DsaKeyPoolCompressionBf16,
    numerical_contract: StableFloat32SoftmaxBf16ProbabilityOrderedFloat32AccumulationV1,
    head_dimension,
    pool_size,
    maximum_tokens,
    pool_capacity,
    function_symbol,
    source,
  }
}
