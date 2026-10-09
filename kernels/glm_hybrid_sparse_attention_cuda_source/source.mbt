///|
fn valid_sparse_attention_symbol(symbol : StringView) -> Bool {
  if symbol.is_empty() || symbol.length() > 128 {
    return false
  }
  symbol[0] is ('a'..='z' | 'A'..='Z' | '_') &&
  symbol.all(c => c is ('a'..='z' | 'A'..='Z' | '0'..='9' | '_'))
}

///|
fn checked_source_product(left : Int64, right : Int64) -> Int64? {
  if left <= 0L || right <= 0L || left > 9223372036854775807L / right {
    None
  } else {
    Some(left * right)
  }
}

///|
/// Renders a deliberately serial correctness oracle over already-projected
/// Q/K/V and one shared selected-index row per query. Invalid or duplicate
/// non-sentinel indices fail the complete query row closed to zero.
pub fn selected_index_sparse_attention_bf16_source(
  query_head_count~ : Int,
  key_value_head_count~ : Int,
  query_key_dimension~ : Int,
  value_dimension~ : Int,
  selected_width~ : Int,
  maximum_tokens~ : Int,
  maximum_queries? : Int = maximum_tokens,
  function_symbol~ : String,
) -> GlmHybridSparseAttentionCudaSource raise GlmHybridSparseAttentionCudaSourceError {
  if query_head_count <= 0 || query_head_count > 1024 {
    raise InvalidQueryHeadCount
  }
  if key_value_head_count <= 0 ||
    key_value_head_count > query_head_count ||
    query_head_count % key_value_head_count != 0 {
    raise InvalidKeyValueHeadCount
  }
  if query_key_dimension <= 0 || query_key_dimension > 65536 {
    raise InvalidQueryKeyDimension
  }
  if value_dimension <= 0 || value_dimension > 65536 {
    raise InvalidValueDimension
  }
  if selected_width <= 0 || selected_width > 65536 {
    raise InvalidSelectedWidth
  }
  if maximum_tokens <= 0 || maximum_tokens > 1048576 {
    raise InvalidMaximumTokens
  }
  if maximum_queries <= 0 || maximum_queries > maximum_tokens {
    raise InvalidMaximumTokens
  }
  if !valid_sparse_attention_symbol(function_symbol) {
    raise InvalidFunctionSymbol
  }
  guard checked_source_product(
      maximum_queries.to_int64(),
      query_head_count.to_int64(),
    )
    is Some(query_heads) else {
    raise SizeOverflow
  }
  guard checked_source_product(query_heads, value_dimension.to_int64())
    is Some(output_elements) else {
    raise SizeOverflow
  }
  if output_elements > 2147483647L {
    raise SizeOverflow
  }
  let source = @utf8.encode(
    "#include <cuda_bf16.h>\n#include <stdint.h>\n#include <math.h>\n" +
    "extern \"C\" __global__ void " +
    function_symbol +
    "(const int32_t *counts, const int32_t *query_positions, const int32_t *key_count, const int32_t *key_valid, const __nv_bfloat16 *queries, const __nv_bfloat16 *keys, const __nv_bfloat16 *values, const int32_t *selected_indices, __nv_bfloat16 *output) {\n" +
    "  if (blockIdx.x != 0 || threadIdx.x != 0) return;\n" +
    "  for (unsigned long long cell = 0; cell < " +
    output_elements.to_string() +
    "ULL; ++cell) output[cell] = __float2bfloat16_rn(0.0f);\n" +
    "  const int query_count = counts[3];\n" +
    "  const int keys_live = *key_count;\n" +
    "  if (counts[2] != 1 || query_count <= 0 || query_count > " +
    maximum_queries.to_string() +
    " || keys_live <= 0 || keys_live > " +
    maximum_tokens.to_string() +
    ") return;\n" +
    "  const float scale = rsqrtf(" +
    query_key_dimension.to_string() +
    ".0f);\n" +
    "  const int query_heads = " +
    query_head_count.to_string() +
    ";\n  const int kv_heads = " +
    key_value_head_count.to_string() +
    ";\n  const int groups = query_heads / kv_heads;\n" +
    "  for (int query = 0; query < query_count; ++query) {\n" +
    "    const int position = query_positions[query];\n" +
    "    if (position < 0 || position >= keys_live) continue;\n" +
    "    const int selected_base = query * " +
    selected_width.to_string() +
    ";\n" +
    "    int row_valid = 1;\n" +
    "    for (int rank = 0; rank < " +
    selected_width.to_string() +
    "; ++rank) {\n" +
    "      const int key = selected_indices[selected_base + rank];\n" +
    "      if (key == -1) continue;\n" +
    "      if (key < 0 || key >= keys_live || key > position || key_valid[key] == 0) { row_valid = 0; break; }\n" +
    "      for (int previous = 0; previous < rank; ++previous) if (selected_indices[selected_base + previous] == key) { row_valid = 0; break; }\n" +
    "      if (!row_valid) break;\n" +
    "    }\n" +
    "    if (!row_valid) continue;\n" +
    "    for (int head = 0; head < query_heads; ++head) {\n" +
    "      const int kv_head = head / groups;\n" +
    "      float maximum = -INFINITY;\n      int visible = 0;\n" +
    "      for (int rank = 0; rank < " +
    selected_width.to_string() +
    "; ++rank) {\n" +
    "        const int key = selected_indices[selected_base + rank];\n" +
    "        if (key < 0 || key >= keys_live || key > position || key_valid[key] == 0) continue;\n" +
    "        float dot = 0.0f;\n" +
    "        const unsigned long long query_base = ((unsigned long long)query * query_heads + head) * " +
    query_key_dimension.to_string() +
    "ULL;\n" +
    "        const unsigned long long key_base = ((unsigned long long)key * kv_heads + kv_head) * " +
    query_key_dimension.to_string() +
    "ULL;\n" +
    "        for (int component = 0; component < " +
    query_key_dimension.to_string() +
    "; ++component) { const float product = __fmul_rn(__bfloat162float(queries[query_base + component]), __bfloat162float(keys[key_base + component])); dot = __fadd_rn(dot, product); }\n" +
    "        const float score = __fmul_rn(dot, scale);\n" +
    "        if (score > maximum) maximum = score;\n        ++visible;\n" +
    "      }\n" +
    "      if (visible == 0) continue;\n" +
    "      float denominator = 0.0f;\n" +
    "      for (int rank = 0; rank < " +
    selected_width.to_string() +
    "; ++rank) {\n" +
    "        const int key = selected_indices[selected_base + rank];\n" +
    "        if (key < 0 || key >= keys_live || key > position || key_valid[key] == 0) continue;\n" +
    "        float dot = 0.0f;\n" +
    "        const unsigned long long query_base = ((unsigned long long)query * query_heads + head) * " +
    query_key_dimension.to_string() +
    "ULL;\n" +
    "        const unsigned long long key_base = ((unsigned long long)key * kv_heads + kv_head) * " +
    query_key_dimension.to_string() +
    "ULL;\n" +
    "        for (int component = 0; component < " +
    query_key_dimension.to_string() +
    "; ++component) { const float product = __fmul_rn(__bfloat162float(queries[query_base + component]), __bfloat162float(keys[key_base + component])); dot = __fadd_rn(dot, product); }\n" +
    "        denominator = __fadd_rn(denominator, expf(__fsub_rn(__fmul_rn(dot, scale), maximum)));\n" +
    "      }\n" +
    "      for (int component = 0; component < " +
    value_dimension.to_string() +
    "; ++component) {\n" +
    "        float accumulator = 0.0f;\n" +
    "        for (int rank = 0; rank < " +
    selected_width.to_string() +
    "; ++rank) {\n" +
    "          const int key = selected_indices[selected_base + rank];\n" +
    "          if (key < 0 || key >= keys_live || key > position || key_valid[key] == 0) continue;\n" +
    "          float dot = 0.0f;\n" +
    "          const unsigned long long query_base = ((unsigned long long)query * query_heads + head) * " +
    query_key_dimension.to_string() +
    "ULL;\n" +
    "          const unsigned long long key_base = ((unsigned long long)key * kv_heads + kv_head) * " +
    query_key_dimension.to_string() +
    "ULL;\n" +
    "          for (int inner = 0; inner < " +
    query_key_dimension.to_string() +
    "; ++inner) { const float product = __fmul_rn(__bfloat162float(queries[query_base + inner]), __bfloat162float(keys[key_base + inner])); dot = __fadd_rn(dot, product); }\n" +
    "          const __nv_bfloat16 probability = __float2bfloat16_rn(__fdividef(expf(__fsub_rn(__fmul_rn(dot, scale), maximum)), denominator));\n" +
    "          const unsigned long long value_base = ((unsigned long long)key * kv_heads + kv_head) * " +
    value_dimension.to_string() +
    "ULL;\n" +
    "          const float product = __fmul_rn(__bfloat162float(probability), __bfloat162float(values[value_base + component]));\n" +
    "          accumulator = __fadd_rn(accumulator, product);\n" +
    "        }\n" +
    "        const unsigned long long output_index = ((unsigned long long)query * query_heads + head) * " +
    value_dimension.to_string() +
    "ULL + component;\n" +
    "        output[output_index] = __float2bfloat16_rn(accumulator);\n" +
    "      }\n" +
    "    }\n  }\n}\n",
  )
  {
    operation: SelectedIndexSparseAttentionBf16,
    numerical_contract: SerialOrderedFloat32QkSoftmaxBf16ProbabilityPvBf16OutputV1,
    query_head_count,
    key_value_head_count,
    query_key_dimension,
    value_dimension,
    selected_width,
    maximum_tokens,
    maximum_queries,
    function_symbol,
    source,
  }
}
