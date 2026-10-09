///|
fn valid_grouped_routing_symbol(value : StringView) -> Bool {
  !value.is_empty() &&
  value.length() <= 128 &&
  value[0] is ('a'..='z' | 'A'..='Z' | '_') &&
  value.all(c => c is ('a'..='z' | 'A'..='Z' | '0'..='9' | '_'))
}

///|
pub fn grouped_choice_and_selected_weight_f32_source(
  routed_expert_count~ : Int,
  group_count~ : Int,
  group_score_topk~ : Int,
  selected_group_count~ : Int,
  experts_per_token~ : Int,
  normalize~ : Bool,
  scaling~ : Double,
  maximum_rows~ : Int,
  function_symbol~ : String,
) -> GroupedRoutingCudaSource raise GroupedRoutingCudaSourceError {
  if routed_expert_count <= 0 ||
    routed_expert_count > 1048576 ||
    experts_per_token <= 0 ||
    experts_per_token > routed_expert_count ||
    experts_per_token > 256 {
    raise InvalidExpertGeometry
  }
  if group_count <= 0 ||
    group_count > 64 ||
    routed_expert_count % group_count != 0 ||
    group_score_topk <= 0 ||
    group_score_topk > routed_expert_count / group_count ||
    group_score_topk > 16 ||
    selected_group_count <= 0 ||
    selected_group_count > group_count ||
    selected_group_count > 16 ||
    selected_group_count * (routed_expert_count / group_count) <
    experts_per_token {
    raise InvalidGroupGeometry
  }
  if maximum_rows <= 0 || maximum_rows > 1048576 {
    raise InvalidMaximumRows
  }
  if scaling.is_nan() || scaling.is_inf() || scaling <= 0.0 {
    raise InvalidScaling
  }
  if !valid_grouped_routing_symbol(function_symbol) {
    raise InvalidFunctionSymbol
  }
  let cells = maximum_rows.to_int64() * routed_expert_count.to_int64()
  if cells <= 0L || cells > 9223372036854775552L {
    raise DimensionOverflow
  }
  let experts_per_group = routed_expert_count / group_count
  let normalize_literal = if normalize { "true" } else { "false" }
  let scaling_literal = @luna_cuda_pointwise_aot.canonical_cuda_float_literal(
    scaling,
  )
  let source = @utf8.encode(
    "#include <math.h>\n#include <stdint.h>\n#include <stdbool.h>\n" +
    "extern \"C\" __global__ void " +
    function_symbol +
    "(const int32_t *counts, const float *uncorrected_scores, const float *choice_scores, int32_t *selected_indices, float *selected_weights) {\n" +
    "  const int32_t live_rows = counts[3]; if (live_rows < 0 || live_rows > " +
    maximum_rows.to_string() +
    ") return;\n" +
    "  const int32_t row = (int32_t)blockIdx.x; if (row >= live_rows || threadIdx.x != 0) return;\n" +
    "  const unsigned long long score_base = (unsigned long long)row * " +
    routed_expert_count.to_string() +
    "ULL; const unsigned long long output_base = (unsigned long long)row * " +
    experts_per_token.to_string() +
    "ULL;\n" +
    "  bool valid = true; for (int32_t expert = 0; expert < " +
    routed_expert_count.to_string() +
    "; ++expert) if (!isfinite(uncorrected_scores[score_base + expert]) || !isfinite(choice_scores[score_base + expert])) valid = false;\n" +
    "  if (!valid) { for (int32_t slot = 0; slot < " +
    experts_per_token.to_string() +
    "; ++slot) { selected_indices[output_base + slot] = -1; selected_weights[output_base + slot] = 0.0f; } return; }\n" +
    "  float group_scores[" +
    group_count.to_string() +
    "];\n" +
    "  for (int32_t group = 0; group < " +
    group_count.to_string() +
    "; ++group) { int32_t group_top[" +
    group_score_topk.to_string() +
    "]; float sum = 0.0f;\n" +
    "    for (int32_t slot = 0; slot < " +
    group_score_topk.to_string() +
    "; ++slot) { int32_t best = -1; float best_value = -INFINITY;\n" +
    "      for (int32_t offset = 0; offset < " +
    experts_per_group.to_string() +
    "; ++offset) { const int32_t expert = group * " +
    experts_per_group.to_string() +
    " + offset; bool used = false; for (int32_t previous = 0; previous < slot; ++previous) if (group_top[previous] == expert) used = true; const float value = choice_scores[score_base + expert]; if (!used && (best < 0 || value > best_value || (value == best_value && expert < best))) { best = expert; best_value = value; } }\n" +
    "      group_top[slot] = best; sum = __fadd_rn(sum, best_value); } group_scores[group] = sum; }\n" +
    "  int32_t selected_groups[" +
    selected_group_count.to_string() +
    "]; for (int32_t slot = 0; slot < " +
    selected_group_count.to_string() +
    "; ++slot) { int32_t best = -1; float best_value = -INFINITY;\n" +
    "    for (int32_t group = 0; group < " +
    group_count.to_string() +
    "; ++group) { bool used = false; for (int32_t previous = 0; previous < slot; ++previous) if (selected_groups[previous] == group) used = true; const float value = group_scores[group]; if (!used && (best < 0 || value > best_value || (value == best_value && group < best))) { best = group; best_value = value; } } selected_groups[slot] = best; }\n" +
    "  for (int32_t slot = 0; slot < " +
    experts_per_token.to_string() +
    "; ++slot) { int32_t best = -1; float best_value = -INFINITY;\n" +
    "    for (int32_t expert = 0; expert < " +
    routed_expert_count.to_string() +
    "; ++expert) { const int32_t group = expert / " +
    experts_per_group.to_string() +
    "; bool group_selected = false; for (int32_t group_slot = 0; group_slot < " +
    selected_group_count.to_string() +
    "; ++group_slot) if (selected_groups[group_slot] == group) group_selected = true; bool used = false; for (int32_t previous = 0; previous < slot; ++previous) if (selected_indices[output_base + previous] == expert) used = true; const float value = choice_scores[score_base + expert]; if (group_selected && !used && (best < 0 || value > best_value || (value == best_value && expert < best))) { best = expert; best_value = value; } } selected_indices[output_base + slot] = best; }\n" +
    "  float denominator = 0.0f; for (int32_t slot = 0; slot < " +
    experts_per_token.to_string() +
    "; ++slot) denominator = __fadd_rn(denominator, uncorrected_scores[score_base + selected_indices[output_base + slot]]); if (" +
    normalize_literal +
    ") denominator = __fadd_rn(denominator, 1.0e-20f);\n" +
    "  for (int32_t slot = 0; slot < " +
    experts_per_token.to_string() +
    "; ++slot) { const float value = uncorrected_scores[score_base + selected_indices[output_base + slot]]; const float normalized = " +
    normalize_literal +
    " ? __fdiv_rn(value, denominator) : value; selected_weights[output_base + slot] = __fmul_rn(normalized, " +
    scaling_literal +
    "); }\n" +
    "}\n",
  )
  {
    operation: GroupedChoiceAndSelectedWeightF32,
    numerical_contract: SerialF32TopGroupChoiceGatherNormalizeScaleV1,
    routed_expert_count,
    group_count,
    group_score_topk,
    selected_group_count,
    experts_per_token,
    normalize,
    scaling,
    maximum_rows,
    function_symbol,
    source,
  }
}
