///|
pub fn device_sampling_cuda_function_symbol() -> String {
  "lunaflux_sample_restrictive_topk_bf16_rows_v2"
}

///|
fn device_sampling_source_text(
  plan : @luna_sampling_strategy.SamplingDevicePlan,
  physical : @elementwise_physical_ir.StochasticProgram,
) -> String {
  let (_, owners, _) = physical.selection().geometry()
  let vocabulary = plan.vocabulary_size().to_string()
  let maximum_rows = plan.maximum_rows().to_string()
  let maximum_top_k = physical.maximum_top_k().to_string()
  let weight_offset = plan.candidate_weight_offset_bytes().to_string()
  "#include <cuda_bf16.h>\n#include <math.h>\n#include <stdint.h>\n#include <limits.h>\n" +
  "struct __align__(8) LunaSamplingParametersV2 { int32_t mode; int32_t top_k; double temperature; double top_p; uint64_t seed; int32_t sample_index; int32_t flags; };\n" +
  "static_assert(sizeof(LunaSamplingParametersV2) == 40, \"sampling parameter ABI\");\n" +
  "__device__ __forceinline__ bool luna_preferred_v2(double a, int ai, double b, int bi) { return a > b || (a == b && ai < bi); }\n" +
  "__device__ __forceinline__ uint64_t luna_counter_v2(uint64_t seed, int sample_index) { uint64_t v = seed + ((uint64_t)sample_index + 1ULL) * 0x9e3779b97f4a7c15ULL; v = (v ^ (v >> 30)) * 0xbf58476d1ce4e5b9ULL; v = (v ^ (v >> 27)) * 0x94d049bb133111ebULL; v ^= v >> 31; return v == 0ULL ? 0x9e3779b97f4a7c15ULL : v; }\n" +
  "extern \"C\" __global__ void lunaflux_sample_restrictive_topk_bf16_rows_v2(" +
  "const int32_t *counts, const int32_t *row_offsets, const __nv_bfloat16 *logits, " +
  "const LunaSamplingParametersV2 *parameters, void *workspace, int32_t *results) {\n" +
  "  __shared__ double reduce_values[\{owners}];\n  __shared__ int reduce_tokens[\{owners}];\n  __shared__ int reduce_invalid[\{owners}];\n" +
  "  __shared__ double selected_value;\n  __shared__ int selected_token;\n  __shared__ int parameter_status;\n" +
  "  const int row = (int)blockIdx.x; const int lane = (int)threadIdx.x;\n" +
  "  const int row_count = counts[2]; const int token_count = counts[3];\n" +
  "  if (row >= row_count || row >= " +
  maximum_rows +
  ") return;\n" +
  "  const int token_end = row_offsets[row + 1];\n" +
  "  if (token_end <= 0 || token_end > token_count) { if (lane == 0) { results[row * 2] = -1; results[row * 2 + 1] = -2; } return; }\n" +
  "  const LunaSamplingParametersV2 p = parameters[row];\n" +
  "  if (lane == 0) { parameter_status = (p.mode != 1 || !(p.flags & 1) || p.top_k <= 0 || p.top_k > " +
  maximum_top_k +
  " || !isfinite(p.temperature) || p.temperature <= 0.0 || p.sample_index < 0 || p.seed == 0ULL || ((p.flags & 2) && (!isfinite(p.top_p) || p.top_p <= 0.0 || p.top_p > 1.0))) ? -3 : -1; }\n" +
  "  __syncthreads(); if (parameter_status != -1) { if (lane == 0) { results[row * 2] = -1; results[row * 2 + 1] = parameter_status; } return; }\n" +
  "  const __nv_bfloat16 *row_logits = logits + (unsigned long long)(token_end - 1) * " +
  vocabulary +
  "ULL;\n" +
  "  int invalid = INT_MAX; for (int token = lane; token < " +
  vocabulary +
  "; token += \{owners}) { const double value = (double)__bfloat162float(row_logits[token]); if (!isfinite(value) && token < invalid) invalid = token; }\n" +
  "  reduce_invalid[lane] = invalid; __syncthreads(); for (int stride = \{owners / 2}; stride > 0; stride >>= 1) { if (lane < stride && reduce_invalid[lane + stride] < reduce_invalid[lane]) reduce_invalid[lane] = reduce_invalid[lane + stride]; __syncthreads(); }\n" +
  "  if (reduce_invalid[0] != INT_MAX) { if (lane == 0) { results[row * 2] = -1; results[row * 2 + 1] = reduce_invalid[0]; } return; }\n" +
  "  int32_t *candidate_tokens = (int32_t *)workspace + row * " +
  maximum_top_k +
  ";\n" +
  "  double *candidate_weights = (double *)((unsigned char *)workspace + " +
  weight_offset +
  ") + row * " +
  maximum_top_k +
  ";\n" +
  "  const int retained_top_k = p.top_k < " +
  vocabulary +
  " ? p.top_k : " +
  vocabulary +
  ";\n" +
  "  if (lane == 0) { selected_value = INFINITY; selected_token = -1; } __syncthreads();\n" +
  "  for (int rank = 0; rank < retained_top_k; ++rank) {\n" +
  "    double best = -INFINITY; int best_token = INT_MAX; const double previous_value = selected_value; const int previous_token = selected_token;\n" +
  "    for (int token = lane; token < " +
  vocabulary +
  "; token += \{owners}) { const double value = (double)__bfloat162float(row_logits[token]); const bool below_previous = rank == 0 || value < previous_value || (value == previous_value && token > previous_token); if (below_previous && luna_preferred_v2(value, token, best, best_token)) { best = value; best_token = token; } }\n" +
  "    reduce_values[lane] = best; reduce_tokens[lane] = best_token; __syncthreads();\n" +
  "    for (int stride = \{owners / 2}; stride > 0; stride >>= 1) { if (lane < stride && luna_preferred_v2(reduce_values[lane + stride], reduce_tokens[lane + stride], reduce_values[lane], reduce_tokens[lane])) { reduce_values[lane] = reduce_values[lane + stride]; reduce_tokens[lane] = reduce_tokens[lane + stride]; } __syncthreads(); }\n" +
  "    if (lane == 0) { selected_value = reduce_values[0]; selected_token = reduce_tokens[0]; candidate_tokens[rank] = selected_token; } __syncthreads();\n" +
  "  }\n" +
  "  if (lane == 0) {\n" +
  stochastic_final_source(physical) +
  "  }\n}\n"
}

///|
fn device_sampling_recipe(
  plan : @luna_sampling_strategy.SamplingDevicePlan,
  target : @catalog.DeviceTarget,
  compiler : @luna_specialization.LunaCompilerPolicy,
  source_sha256 : String,
) -> Bytes {
  let identity = compiler.identity()
  @utf8.encode(
    "schema=lunaflux-restrictive-topk-bf16-sampling-aot-v2\n" +
    "function=" +
    device_sampling_cuda_function_symbol() +
    "\nsource_sha256=" +
    source_sha256 +
    "\ntoolchain_sha256=" +
    identity.toolchain_digest_sha256() +
    "\ntarget=sm_" +
    target.compute_major().to_string() +
    target.compute_minor().to_string() +
    "\nvocabulary_size=" +
    plan.vocabulary_size().to_string() +
    "\nmaximum_rows=" +
    plan.maximum_rows().to_string() +
    "\nmaximum_top_k=" +
    plan.maximum_top_k().to_string() +
    "\nparameter_row_bytes=" +
    plan.parameter_row_bytes().to_string() +
    "\nresult_row_bytes=" +
    plan.result_row_bytes().to_string() +
    "\ncandidate_weight_offset_bytes=" +
    plan.candidate_weight_offset_bytes().to_string() +
    "\nworkspace_bytes=" +
    plan.workspace_bytes().to_string() +
    "\nblock=256,1,1\n" +
    "numerical_contract=finite_bf16_total_order_f64_softmax_counter_v2\n" +
    "candidate_contract=explicit_restrictive_top_k_top_p_prefix_v1\n" +
    "reassociate=false\nmanifest_bindable=false\noutput=cubin\n",
  )
}

///|
/// Emits a bounded standalone CUDA candidate. Compilation remains offline and
/// the result cannot enter production until the paged manifest owns the six
/// exact operands of this ABI.
pub fn lower_device_sampling_cuda_aot(
  plan : @luna_sampling_strategy.SamplingDevicePlan,
  target : @catalog.DeviceTarget,
  compiler : @luna_specialization.LunaCompilerPolicy,
  limits : DeviceSamplingAotLimits,
) -> DeviceSamplingCudaLowering raise SamplingBoundaryError {
  if plan.maximum_rows() <= 0 ||
    plan.vocabulary_size() <= 0 ||
    plan.maximum_top_k() <= 0 ||
    plan.parameter_row_bytes() != 40L ||
    plan.result_row_bytes() != 8L ||
    plan.workspace_bytes() <= 0L {
    raise Invalid(DeviceSamplingPlan)
  }
  if !target.supports_bf16() || compiler.reassociate() {
    raise Unsupported(ArtifactTarget)
  }
  let physical_program = stochastic_physical_program(plan)
  let source = @utf8.encode(device_sampling_source_text(plan, physical_program))
  if source.length() > limits.max_source_bytes {
    raise LimitExceeded(ArtifactSource)
  }
  let source_sha256 = sampling_sha256(source)
  let recipe = device_sampling_recipe(plan, target, compiler, source_sha256)
  if recipe.length() > limits.max_recipe_bytes {
    raise LimitExceeded(ArtifactRecipe)
  }
  {
    physical_program,
    target,
    plan,
    function_symbol: device_sampling_cuda_function_symbol(),
    source,
    source_sha256,
    recipe,
    recipe_sha256: sampling_sha256(recipe),
    compiler,
  }
}
