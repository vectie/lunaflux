///|
const GREEDY_RESULT_ROW_BYTES : Int64 = 8L

///|
fn sampling_sha256(bytes : Bytes) -> String {
  @crypto.bytes_to_hex_string(@crypto.sha256(bytes))
}

///|
fn sampling_is_sha256(value : StringView) -> Bool {
  if value.length() != 64 {
    return false
  }
  for unit in value {
    if !(unit is ('0'..='9' | 'a'..='f')) {
      return false
    }
  }
  true
}

///|
fn greedy_sampling_source_text(
  vocabulary_size : Int,
  maximum_rows : Int,
) -> String {
  "#include <cuda_bf16.h>\n#include <math.h>\n#include <stdint.h>\n#include <limits.h>\n" +
  "extern \"C\" __global__ void lunaflux_greedy_bf16_rows_v1(" +
  "const int32_t *counts, const int32_t *row_offsets, " +
  "const __nv_bfloat16 *logits, int32_t *results) {\n" +
  "  __shared__ float values[256];\n  __shared__ int tokens[256];\n" +
  "  __shared__ int invalids[256];\n" +
  "  const int row = (int)blockIdx.x;\n  const int lane = (int)threadIdx.x;\n" +
  "  const int row_count = counts[2];\n  const int token_count = counts[3];\n" +
  "  if (row >= row_count || row >= " +
  maximum_rows.to_string() +
  ") return;\n" +
  "  const int token_end = row_offsets[row + 1];\n" +
  "  if (token_end <= 0 || token_end > token_count) { if (lane == 0) { results[row * 2] = -1; results[row * 2 + 1] = -2; } return; }\n" +
  "  const __nv_bfloat16 *row_logits = logits + (unsigned long long)(token_end - 1) * " +
  vocabulary_size.to_string() +
  "ULL;\n" +
  "  float best = -INFINITY;\n  int best_token = INT_MAX;\n  int invalid = INT_MAX;\n" +
  "  for (int token = lane; token < " +
  vocabulary_size.to_string() +
  "; token += 256) {\n" +
  "    const float value = __bfloat162float(row_logits[token]);\n" +
  "    if (!isfinite(value)) { if (token < invalid) invalid = token; }\n" +
  "    else if (value > best || (value == best && token < best_token)) { best = value; best_token = token; }\n  }\n" +
  "  values[lane] = best; tokens[lane] = best_token; invalids[lane] = invalid;\n  __syncthreads();\n" +
  "  for (int stride = 128; stride > 0; stride >>= 1) {\n" +
  "    if (lane < stride) {\n      const float other = values[lane + stride];\n      const int other_token = tokens[lane + stride];\n" +
  "      if (other > values[lane] || (other == values[lane] && other_token < tokens[lane])) { values[lane] = other; tokens[lane] = other_token; }\n" +
  "      if (invalids[lane + stride] < invalids[lane]) invalids[lane] = invalids[lane + stride];\n    }\n    __syncthreads();\n  }\n" +
  "  if (lane == 0) { const int bad = invalids[0]; results[row * 2] = bad == INT_MAX ? tokens[0] : -1; results[row * 2 + 1] = bad == INT_MAX ? -1 : bad; }\n}\n"
}

///|
/// Returns the fixed product-owned reducer symbol embedded into an approved
/// language-model-head module by the BF16 release producer.
pub fn greedy_sampling_cuda_function_symbol() -> String {
  "lunaflux_greedy_bf16_rows_v1"
}

///|
/// Emits the deterministic reducer translation-unit fragment used when the
/// production BF16 exporter co-compiles greedy selection with the final
/// language-model-head kernel. This remains offline source construction: it
/// grants no compiler, artifact, module, device, or execution authority.
pub fn embedded_greedy_sampling_cuda_source(
  vocabulary_size : Int,
  maximum_rows : Int,
) -> String raise SamplingBoundaryError {
  if vocabulary_size <= 0 {
    raise Invalid(VocabularySize)
  }
  if maximum_rows <= 0 {
    raise Invalid(MaximumRows)
  }
  greedy_sampling_source_text(vocabulary_size, maximum_rows)
}

///|
fn greedy_sampling_recipe(
  target : @catalog.DeviceTarget,
  compiler : @luna_specialization.LunaCompilerPolicy,
  vocabulary_size : Int,
  maximum_rows : Int,
  source_sha256 : String,
) -> Bytes {
  let identity = compiler.identity()
  @utf8.encode(
    "schema=lunaflux-greedy-bf16-sampling-aot-v1\n" +
    "function=lunaflux_greedy_bf16_rows_v1\nsource_sha256=" +
    source_sha256 +
    "\ntoolchain_sha256=" +
    identity.toolchain_digest_sha256() +
    "\ntarget=sm_" +
    target.compute_major().to_string() +
    target.compute_minor().to_string() +
    "\nvocabulary_size=" +
    vocabulary_size.to_string() +
    "\nmaximum_rows=" +
    maximum_rows.to_string() +
    "\nblock=256,1,1\nresult_row_bytes=8\n" +
    "numerical_contract=finite_bf16_argmax_lowest_token_v1\n" +
    "reassociate=false\noutput=cubin\n",
  )
}

///|
/// Emits the closed-source AOT greedy reducer for one exact target and shape.
pub fn lower_greedy_sampling_cuda_aot(
  boundary : HostSamplingBoundary,
  target : @catalog.DeviceTarget,
  compiler : @luna_specialization.LunaCompilerPolicy,
  maximum_rows : Int,
  limits : GreedySamplingAotLimits,
) -> GreedySamplingCudaLowering raise SamplingBoundaryError {
  if maximum_rows <= 0 {
    raise Invalid(MaximumRows)
  }
  if !target.supports_bf16() || compiler.reassociate() {
    raise Unsupported(ArtifactTarget)
  }
  let source = @utf8.encode(
    embedded_greedy_sampling_cuda_source(boundary.vocabulary_size, maximum_rows),
  )
  if source.length() > limits.max_source_bytes {
    raise LimitExceeded(ArtifactSource)
  }
  let source_sha256 = sampling_sha256(source)
  let recipe = greedy_sampling_recipe(
    target,
    compiler,
    boundary.vocabulary_size,
    maximum_rows,
    source_sha256,
  )
  if recipe.length() > limits.max_recipe_bytes {
    raise LimitExceeded(ArtifactRecipe)
  }
  {
    target,
    vocabulary_size: boundary.vocabulary_size,
    maximum_rows,
    function_symbol: "lunaflux_greedy_bf16_rows_v1",
    source,
    source_sha256,
    recipe,
    recipe_sha256: sampling_sha256(recipe),
    compiler,
  }
}
