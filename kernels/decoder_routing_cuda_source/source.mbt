///|
fn valid_routing_symbol(symbol : StringView) -> Bool {
  if symbol.is_empty() || symbol.length() > 128 {
    return false
  }
  symbol[0] is ('a'..='z' | 'A'..='Z' | '_') &&
  symbol.all(c => c is ('a'..='z' | 'A'..='Z' | '0'..='9' | '_'))
}

///|
fn validate_routing_geometry(
  routed_expert_count : Int,
  experts_per_token : Int,
  maximum_rows : Int,
) -> Unit raise DecoderRoutingCudaSourceError {
  if routed_expert_count <= 0 {
    raise InvalidRoutedExpertCount
  }
  if experts_per_token <= 0 ||
    experts_per_token > routed_expert_count ||
    experts_per_token > 256 {
    raise InvalidExpertsPerToken
  }
  if maximum_rows <= 0 || maximum_rows > 1048576 {
    raise InvalidMaximumRows
  }
}

///|
/// Emits deterministic biased top-k selection. Equal finite choice scores are
/// ordered by lower expert id. Non-finite rows emit `-1` indices.
pub fn stable_biased_top_k_i32_source(
  routed_expert_count~ : Int,
  experts_per_token~ : Int,
  maximum_rows~ : Int,
  function_symbol~ : String,
) -> DecoderRoutingCudaSource raise DecoderRoutingCudaSourceError {
  validate_routing_geometry(
    routed_expert_count, experts_per_token, maximum_rows,
  )
  if !valid_routing_symbol(function_symbol) {
    raise InvalidFunctionSymbol
  }
  let source = @utf8.encode(
    "#include <math.h>\n#include <stdint.h>\n" +
    "extern \"C\" __global__ void " +
    function_symbol +
    "(const int32_t *counts, const float *scores, const float *selection_bias, int32_t *selected_indices) {\n" +
    "  const int32_t live_rows = counts[3];\n" +
    "  if (live_rows < 0 || live_rows > " +
    maximum_rows.to_string() +
    ") return;\n" +
    "  const int32_t row = (int32_t)blockIdx.x;\n" +
    "  if (row >= live_rows || threadIdx.x != 0) return;\n" +
    "  const unsigned long long score_base = (unsigned long long)row * " +
    routed_expert_count.to_string() +
    "ULL;\n" +
    "  const unsigned long long output_base = (unsigned long long)row * " +
    experts_per_token.to_string() +
    "ULL;\n" +
    "  bool valid = true;\n" +
    "  for (int32_t expert = 0; expert < " +
    routed_expert_count.to_string() +
    "; ++expert) {\n" +
    "    const float choice = __fadd_rn(scores[score_base + (unsigned long long)expert], selection_bias[expert]);\n" +
    "    if (!isfinite(choice)) valid = false;\n" +
    "  }\n" +
    "  if (!valid) { for (int32_t slot = 0; slot < " +
    experts_per_token.to_string() +
    "; ++slot) selected_indices[output_base + (unsigned long long)slot] = -1; return; }\n" +
    "  for (int32_t slot = 0; slot < " +
    experts_per_token.to_string() +
    "; ++slot) {\n" +
    "    int32_t best = -1; float best_value = -INFINITY;\n" +
    "    for (int32_t expert = 0; expert < " +
    routed_expert_count.to_string() +
    "; ++expert) {\n" +
    "      bool used = false; for (int32_t previous = 0; previous < slot; ++previous) if (selected_indices[output_base + (unsigned long long)previous] == expert) used = true;\n" +
    "      if (used) continue;\n" +
    "      const float choice = __fadd_rn(scores[score_base + (unsigned long long)expert], selection_bias[expert]);\n" +
    "      if (best < 0 || choice > best_value || (choice == best_value && expert < best)) { best = expert; best_value = choice; }\n" +
    "    }\n" +
    "    selected_indices[output_base + (unsigned long long)slot] = best;\n" +
    "  }\n" +
    "}\n",
  )
  {
    operation: StableBiasedTopKI32,
    numerical_contract: StableLowerExpertIdTopKV1,
    routed_expert_count,
    experts_per_token,
    maximum_rows,
    function_symbol,
    source,
  }
}

///|
/// Emits selected-score gathering followed by optional ordered F32
/// normalization and scaling. Indices may come from stable top-k or a token
/// table. Invalid rows emit zero weights.
pub fn selected_expert_weight_finalize_f32_source(
  routed_expert_count~ : Int,
  experts_per_token~ : Int,
  maximum_rows~ : Int,
  normalize~ : Bool,
  scaling~ : Double,
  function_symbol~ : String,
) -> DecoderRoutingCudaSource raise DecoderRoutingCudaSourceError {
  validate_routing_geometry(
    routed_expert_count, experts_per_token, maximum_rows,
  )
  if scaling.is_nan() || scaling.is_inf() || scaling <= 0.0 {
    raise InvalidScaling
  }
  if !valid_routing_symbol(function_symbol) {
    raise InvalidFunctionSymbol
  }
  let scaling_literal = @luna_cuda_pointwise_aot.canonical_cuda_float_literal(
    scaling,
  )
  let normalize_literal = if normalize { "true" } else { "false" }
  let source = @utf8.encode(
    "#include <math.h>\n#include <stdint.h>\n" +
    "extern \"C\" __global__ void " +
    function_symbol +
    "(const int32_t *counts, const float *scores, const int32_t *selected_indices, float *selected_weights) {\n" +
    "  const int32_t live_rows = counts[3];\n" +
    "  if (live_rows < 0 || live_rows > " +
    maximum_rows.to_string() +
    ") return;\n" +
    "  const int32_t row = (int32_t)blockIdx.x;\n" +
    "  if (row >= live_rows || threadIdx.x != 0) return;\n" +
    "  const unsigned long long score_base = (unsigned long long)row * " +
    routed_expert_count.to_string() +
    "ULL;\n" +
    "  const unsigned long long selected_base = (unsigned long long)row * " +
    experts_per_token.to_string() +
    "ULL;\n" +
    "  bool valid = true; float sum = 0.0f;\n" +
    "  for (int32_t slot = 0; slot < " +
    experts_per_token.to_string() +
    "; ++slot) {\n" +
    "    const int32_t expert = selected_indices[selected_base + (unsigned long long)slot];\n" +
    "    if (expert < 0 || expert >= " +
    routed_expert_count.to_string() +
    ") { valid = false; continue; }\n" +
    "    for (int32_t previous = 0; previous < slot; ++previous) if (selected_indices[selected_base + (unsigned long long)previous] == expert) valid = false;\n" +
    "    const float value = scores[score_base + (unsigned long long)expert];\n" +
    "    if (!isfinite(value)) valid = false;\n" +
    "    if (" +
    normalize_literal +
    ") sum = __fadd_rn(sum, value);\n" +
    "  }\n" +
    "  if (" +
    normalize_literal +
    " && (!isfinite(sum) || sum <= 0.0f)) valid = false;\n" +
    "  if (!valid) { for (int32_t slot = 0; slot < " +
    experts_per_token.to_string() +
    "; ++slot) selected_weights[selected_base + (unsigned long long)slot] = 0.0f; return; }\n" +
    "  for (int32_t slot = 0; slot < " +
    experts_per_token.to_string() +
    "; ++slot) {\n" +
    "    const int32_t expert = selected_indices[selected_base + (unsigned long long)slot];\n" +
    "    const float value = scores[score_base + (unsigned long long)expert];\n" +
    "    const float normalized = " +
    normalize_literal +
    " ? __fdiv_rn(value, sum) : value;\n" +
    "    selected_weights[selected_base + (unsigned long long)slot] = __fmul_rn(normalized, " +
    scaling_literal +
    ");\n" +
    "  }\n" +
    "}\n",
  )
  {
    operation: SelectedExpertWeightFinalizeF32,
    numerical_contract: OrderedF32NormalizeScaleV1,
    routed_expert_count,
    experts_per_token,
    maximum_rows,
    function_symbol,
    source,
  }
}
