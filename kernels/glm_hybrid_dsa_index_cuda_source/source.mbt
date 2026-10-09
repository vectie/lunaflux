///|
fn valid_dsa_index_symbol(symbol : StringView) -> Bool {
  if symbol.is_empty() || symbol.length() > 128 {
    return false
  }
  symbol[0] is ('a'..='z' | 'A'..='Z' | '_') &&
  symbol.all(c => c is ('a'..='z' | 'A'..='Z' | '0'..='9' | '_'))
}

///|
/// Renders a deliberately serial correctness oracle for projected DSA scores,
/// causal candidate filtering, deterministic top-k, pool expansion, and the
/// optional visible incomplete tail. It is not a throughput kernel.
pub fn projected_index_score_stable_topk_bf16_source(
  head_count~ : Int,
  head_dimension~ : Int,
  top_k~ : Int,
  pool_size~ : Int,
  append_incomplete_tail~ : Bool,
  maximum_tokens~ : Int,
  maximum_queries? : Int = maximum_tokens,
  function_symbol~ : String,
) -> GlmHybridDsaIndexCudaSource raise GlmHybridDsaIndexCudaSourceError {
  if head_count <= 0 || head_count > 1024 {
    raise InvalidHeadCount
  }
  if head_dimension <= 0 || head_dimension > 65536 {
    raise InvalidHeadDimension
  }
  if top_k <= 0 || top_k > 65536 {
    raise InvalidTopK
  }
  if pool_size <= 0 || pool_size > 1024 || top_k % pool_size != 0 {
    raise InvalidPoolSize
  }
  if maximum_tokens < pool_size || maximum_tokens > 1048576 {
    raise InvalidMaximumTokens
  }
  if maximum_queries <= 0 || maximum_queries > maximum_tokens {
    raise InvalidMaximumTokens
  }
  if !valid_dsa_index_symbol(function_symbol) {
    raise InvalidFunctionSymbol
  }
  let candidate_capacity = maximum_tokens / pool_size
  let output_width = top_k +
    (if append_incomplete_tail { pool_size - 1 } else { 0 })
  if output_width <= 0 ||
    maximum_queries.to_int64() > 2147483647L / output_width.to_int64() {
    raise SizeOverflow
  }
  let tail_source = if append_incomplete_tail {
    "    int first_valid = " +
    maximum_tokens.to_string() +
    ";\n" +
    "    int visible_count = 0;\n    int tail_layout_valid = 1;\n    int saw_valid = 0;\n" +
    "    for (int key = 0; key < " +
    maximum_tokens.to_string() +
    "; ++key) {\n" +
    "      if (key_valid[key] != 0) { if (first_valid == " +
    maximum_tokens.to_string() +
    ") first_valid = key; if (key <= query_position) { saw_valid = 1; ++visible_count; } } else if (saw_valid && key <= query_position) { tail_layout_valid = 0; }\n" +
    "    }\n" +
    "    const int tail_count = visible_count % " +
    pool_size.to_string() +
    ";\n" +
    "    const int tail_start = first_valid + visible_count - tail_count;\n" +
    "    const int tail_base = selection_limit * " +
    pool_size.to_string() +
    ";\n" +
    "    for (int offset = 0; offset < " +
    (pool_size - 1).to_string() +
    "; ++offset) {\n" +
    "      const int raw = tail_start + offset;\n" +
    "      if (tail_layout_valid && offset < tail_count && raw >= 0 && raw < " +
    maximum_tokens.to_string() +
    " && raw <= query_position && key_valid[raw] != 0) selected_indices[output_base + tail_base + offset] = raw;\n" +
    "    }\n"
  } else {
    ""
  }
  let source = @utf8.encode(
    "#include <cuda_bf16.h>\n#include <stdint.h>\n#include <math.h>\n" +
    "extern \"C\" __global__ void " +
    function_symbol +
    "(const int32_t *counts, const int32_t *query_positions, const __nv_bfloat16 *queries, const __nv_bfloat16 *head_weights, const __nv_bfloat16 *candidate_keys, const int32_t *candidate_raw_indices, const int32_t *candidate_valid, const int32_t *candidate_count, const int32_t *key_valid, int32_t *selected_indices) {\n" +
    "  if (blockIdx.x != 0 || threadIdx.x != 0) return;\n" +
    "  const int query_count = counts[3];\n" +
    "  const int output_width = " +
    output_width.to_string() +
    ";\n" +
    "  for (unsigned long long cell = 0; cell < " +
    maximum_queries.to_string() +
    "ULL * (unsigned long long)output_width; ++cell) selected_indices[cell] = -1;\n" +
    "  if (counts[2] != 1 || query_count <= 0 || query_count > " +
    maximum_queries.to_string() +
    ") return;\n" +
    "  const int candidates = *candidate_count;\n" +
    "  if (candidates < 0 || candidates > " +
    candidate_capacity.to_string() +
    ") return;\n" +
    "  const float dimension_scale = rsqrtf(" +
    head_dimension.to_string() +
    ".0f);\n" +
    "  const float head_scale = rsqrtf(" +
    head_count.to_string() +
    ".0f);\n" +
    "  for (int query = 0; query < query_count; ++query) {\n" +
    "    const int query_position = query_positions[query];\n" +
    "    const int output_base = query * output_width;\n" +
    "    if (query_position < 0 || query_position >= " +
    maximum_tokens.to_string() +
    ") continue;\n" +
    "    int32_t selected_candidates[" +
    (top_k / pool_size).to_string() +
    "];\n" +
    "    for (int rank = 0; rank < " +
    (top_k / pool_size).to_string() +
    "; ++rank) selected_candidates[rank] = -1;\n" +
    "    const int selection_limit = candidates < " +
    (top_k / pool_size).to_string() +
    " ? candidates : " +
    (top_k / pool_size).to_string() +
    ";\n" +
    "    for (int rank = 0; rank < selection_limit; ++rank) {\n" +
    "      int best = -1;\n      float best_score = -INFINITY;\n" +
    "      for (int candidate = 0; candidate < candidates; ++candidate) {\n" +
    "        int already_selected = 0;\n" +
    "        for (int previous = 0; previous < rank; ++previous) already_selected |= selected_candidates[previous] == candidate;\n" +
    "        const int final_raw = candidate_raw_indices[candidate * " +
    pool_size.to_string() +
    " + " +
    (pool_size - 1).to_string() +
    "];\n" +
    "        if (already_selected || candidate_valid[candidate] == 0 || final_raw < 0 || final_raw > query_position) continue;\n" +
    "        float score = 0.0f;\n" +
    "        for (int head = 0; head < " +
    head_count.to_string() +
    "; ++head) {\n" +
    "          float dot = 0.0f;\n" +
    "          const int query_base = (query * " +
    head_count.to_string() +
    " + head) * " +
    head_dimension.to_string() +
    ";\n" +
    "          const int key_base = candidate * " +
    head_dimension.to_string() +
    ";\n" +
    "          for (int component = 0; component < " +
    head_dimension.to_string() +
    "; ++component) {\n" +
    "            const float product = __fmul_rn(__bfloat162float(queries[query_base + component]), __bfloat162float(candidate_keys[key_base + component]));\n" +
    "            dot = __fadd_rn(dot, product);\n" +
    "          }\n" +
    "          const float activated = fmaxf(__fmul_rn(dot, dimension_scale), 0.0f);\n" +
    "          const float weighted = __fmul_rn(activated, __fmul_rn(__bfloat162float(head_weights[query * " +
    head_count.to_string() +
    " + head]), head_scale));\n" +
    "          score = __fadd_rn(score, weighted);\n" +
    "        }\n" +
    "        if (score > best_score || (score == best_score && (best < 0 || candidate < best))) { best = candidate; best_score = score; }\n" +
    "      }\n" +
    "      if (best < 0) break;\n" +
    "      selected_candidates[rank] = best;\n" +
    "      for (int offset = 0; offset < " +
    pool_size.to_string() +
    "; ++offset) selected_indices[output_base + rank * " +
    pool_size.to_string() +
    " + offset] = candidate_raw_indices[best * " +
    pool_size.to_string() +
    " + offset];\n" +
    "    }\n" +
    tail_source +
    "  }\n}\n",
  )
  {
    operation: ProjectedIndexScoreStableTopK,
    numerical_contract: SerialOrderedFloat32ReluWeightedScoreStableTopKV1,
    head_count,
    head_dimension,
    top_k,
    pool_size,
    maximum_tokens,
    maximum_queries,
    candidate_capacity,
    output_width,
    function_symbol,
    source,
  }
}

///|
pub fn reuse_selected_indices_i32_source(
  output_width~ : Int,
  maximum_tokens~ : Int,
  function_symbol~ : String,
) -> GlmHybridDsaIndexCudaSource raise GlmHybridDsaIndexCudaSourceError {
  if output_width <= 0 || output_width > 65536 {
    raise InvalidOutputWidth
  }
  if maximum_tokens <= 0 || maximum_tokens > 1048576 {
    raise InvalidMaximumTokens
  }
  if !valid_dsa_index_symbol(function_symbol) {
    raise InvalidFunctionSymbol
  }
  let source = @utf8.encode(
    "#include <stdint.h>\nextern \"C\" __global__ void " +
    function_symbol +
    "(const int32_t *counts, const int32_t *previous_indices, int32_t *reused_indices) {\n" +
    "  const int query_count = counts[3];\n" +
    "  if (query_count <= 0 || query_count > " +
    maximum_tokens.to_string() +
    ") return;\n" +
    "  const unsigned long long cell = (unsigned long long)blockIdx.x * 256ULL + threadIdx.x;\n" +
    "  const unsigned long long live = (unsigned long long)query_count * " +
    output_width.to_string() +
    "ULL;\n" +
    "  if (cell < live) reused_indices[cell] = previous_indices[cell];\n}\n",
  )
  {
    operation: ReuseSelectedIndicesI32,
    numerical_contract: BitExactI32Copy,
    head_count: 0,
    head_dimension: 0,
    top_k: output_width,
    pool_size: 1,
    maximum_tokens,
    maximum_queries: maximum_tokens,
    candidate_capacity: 0,
    output_width,
    function_symbol,
    source,
  }
}
