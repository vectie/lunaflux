///|
fn qwen_step_rotary_cache_plan(
  profile : @launch_contract.PagedKernelProfile,
  head_dimension : Int,
) -> @elementwise_physical_ir.StepRotaryCachePlan {
  @elementwise_physical_ir.StepRotaryCachePlan::new(
    maximum_tokens=profile.max_query_tokens(),
    head_dimension~,
  ).unwrap()
}

///|
fn qwen_step_rotary_preparation_source(
  consumer_symbol : String,
  numerics : @luna_projection_tile_compiler.AttentionIngressNumerics,
  profile : @launch_contract.PagedKernelProfile,
) -> String {
  let (dimension, _) = numerics.shape()
  let cache = qwen_step_rotary_cache_plan(profile, dimension)
  let (tokens, pairs) = cache.geometry()
  let out = StringBuilder()
  out <+ "static __device__ const float LF_STEP_ROPE_INV_FREQ[\{pairs}] = {"
  for index, frequency in numerics.rotary_basis() {
    if index > 0 {
      out.write_string(", ")
    }
    out.write_string(
      @luna_cuda_pointwise_aot.canonical_cuda_float_literal(
        frequency.to_double(),
      ),
    )
  }
  out <+ "};\n"
  out <+
    "extern \"C\" __global__ void \{consumer_symbol}_rotary_prepare(const int32_t* __restrict__ step_counts, const int32_t* __restrict__ query_positions, float2* __restrict__ rotary_pairs) {\n"
  out <+ "  const int tokens = step_counts[3];\n"
  out <+
    "  if (tokens <= 0 || tokens > \{tokens} || blockDim.x != 128 || blockIdx.y != 0 || blockIdx.z != 0) return;\n"
  out <+
    "  const unsigned long long cell = (unsigned long long)blockIdx.x * 128 + threadIdx.x;\n"
  out <+ "  if (cell >= (unsigned long long)tokens * \{pairs}) return;\n"
  out <+
    "  const int token = (int)(cell / \{pairs}), pair = (int)(cell % \{pairs});\n"
  out <+ "  float sine, cosine;\n"
  out <+
    "  sincosf(__fmul_rn((float)query_positions[token], LF_STEP_ROPE_INV_FREQ[pair]), &sine, &cosine);\n"
  out <+ "  rotary_pairs[cell] = make_float2(sine, cosine);\n}\n"
  out.to_string()
}

///|
fn qwen_step_rotary_recipe(
  symbol : String,
  profile : @launch_contract.PagedKernelProfile,
  head_dimension : Int,
) -> String {
  let cache = qwen_step_rotary_cache_plan(profile, head_dimension)
  let (tokens, pairs) = cache.geometry()
  "rotary_cache_policy=step-prepared-f32-pairs-v1\n" +
  "rotary_prepare_symbol=\{symbol}_rotary_prepare\n" +
  "rotary_cache_bytes=\{cache.byte_count()}\n" +
  "rotary_prepare_abi=step_counts,query_positions,rotary_pairs\n" +
  "rotary_prepare_block=128,1,1\n" +
  "rotary_prepare_grid=\{(tokens.to_int64() * pairs.to_int64() + 127L) / 128L},1,1\n" +
  "rotary_prepare_effect=produce-before-all-layer-consumers\n"
}
