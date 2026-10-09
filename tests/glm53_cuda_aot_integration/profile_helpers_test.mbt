///|
fn full_metadata() -> @glm53_spec.Glm53FullMetadata {
  @glm53_spec.Glm53FullMetadata::from_verified_content(
    @glm53_spec.Glm53FullSpec::official(),
    try! @spec.ContentDigest::from_sha256(String::make(64, 'a')),
  )
}

///|
fn flash_metadata() -> @glm53_spec.Glm53FlashMetadata {
  @glm53_spec.Glm53FlashMetadata::from_verified_content(
    @glm53_spec.Glm53FlashSpec::official(),
    try! @spec.ContentDigest::from_sha256(String::make(64, 'b')),
  )
}

///|
fn target() -> @catalog.DeviceTarget {
  try! @catalog.DeviceTarget::new(
    compute_major=9,
    compute_minor=0,
    supports_bf16=true,
    supports_cublas_lt=true,
  )
}

///|
fn compiler() -> @luna_specialization.LunaCompilerPolicy {
  let identity = try! @luna_specialization.LunaCompilerIdentity::new(
    toolchain_digest_sha256=String::make(64, 'c'),
    major=12,
    minor=9,
    patch=1,
  )
  try! @luna_specialization.LunaCompilerPolicy::new(
    identity~,
    optimization_level=3,
    fused_multiply_add=false,
    reassociate=false,
    max_registers=128,
  )
}

///|
fn limits() -> @glm_hybrid_cuda_aot.GlmHybridCudaAotLimits {
  try! @glm_hybrid_cuda_aot.GlmHybridCudaAotLimits::new(
    maximum_tokens=4096,
    maximum_source_bytes=65536,
    maximum_recipe_bytes=65536,
  )
}

///|
fn find_head_ordinal(
  requirements : @glm_hybrid_kernel_requirements.GlmHybridKernelRequirements,
) -> Int {
  for ordinal, requirement in requirements.requirements() {
    if requirement.operation() is LanguageModelHead(_, _) {
      return ordinal
    }
  }
  abort("missing language-model-head requirement")
}

///|
fn find_attention_output_ordinal(
  requirements : @glm_hybrid_kernel_requirements.GlmHybridKernelRequirements,
) -> Int {
  for ordinal, requirement in requirements.requirements() {
    if requirement.operation() is AttentionOutputProjection(_, _) {
      return ordinal
    }
  }
  abort("missing attention-output projection requirement")
}

///|
fn find_hyper_function_ordinal(
  requirements : @glm_hybrid_kernel_requirements.GlmHybridKernelRequirements,
) -> Int {
  for ordinal, requirement in requirements.requirements() {
    if requirement.operation() is HyperConnectionFunction(_, _, _, _) {
      return ordinal
    }
  }
  abort("missing hyper-connection function")
}

///|
fn find_hyper_sinkhorn_ordinal(
  requirements : @glm_hybrid_kernel_requirements.GlmHybridKernelRequirements,
) -> Int {
  for ordinal, requirement in requirements.requirements() {
    if requirement.operation() is HyperConnectionSinkhorn(_, _, _) {
      return ordinal
    }
  }
  abort("missing hyper-connection Sinkhorn")
}

///|
fn find_dsa_pooling_ordinal(
  requirements : @glm_hybrid_kernel_requirements.GlmHybridKernelRequirements,
) -> Int? {
  for ordinal, requirement in requirements.requirements() {
    if requirement.operation() is DsaIndexPooling(_, _, _) {
      return Some(ordinal)
    }
  }
  None
}

///|
fn find_dsa_index_compute_ordinal(
  requirements : @glm_hybrid_kernel_requirements.GlmHybridKernelRequirements,
) -> Int {
  for ordinal, requirement in requirements.requirements() {
    if requirement.operation() is DsaIndexCompute(_, _, _, _) {
      return ordinal
    }
  }
  abort("missing DSA index-compute requirement")
}

///|
fn find_dsa_reuse_ordinal(
  requirements : @glm_hybrid_kernel_requirements.GlmHybridKernelRequirements,
) -> Int? {
  for ordinal, requirement in requirements.requirements() {
    if requirement.operation() is DsaIndexReuse {
      return Some(ordinal)
    }
  }
  None
}

///|
fn find_dsa_sparse_attention_ordinal(
  requirements : @glm_hybrid_kernel_requirements.GlmHybridKernelRequirements,
) -> Int {
  for ordinal, requirement in requirements.requirements() {
    if requirement.operation() is DsaSparseAttention(_, _, _, _) {
      return ordinal
    }
  }
  abort("missing DSA sparse-attention requirement")
}

///|
fn find_query_a_ordinal(
  requirements : @glm_hybrid_kernel_requirements.GlmHybridKernelRequirements,
) -> Int {
  for ordinal, requirement in requirements.requirements() {
    if requirement.operation() is DsaQueryAProjection(_, _) {
      return ordinal
    }
  }
  abort("missing DSA q-a projection")
}

///|
fn find_kv_a_ordinal(
  requirements : @glm_hybrid_kernel_requirements.GlmHybridKernelRequirements,
) -> Int {
  for ordinal, requirement in requirements.requirements() {
    if requirement.operation() is DsaKeyValueAProjection(_, _, _) {
      return ordinal
    }
  }
  abort("missing DSA kv-a projection")
}

///|
fn find_query_a_norm_ordinal(
  requirements : @glm_hybrid_kernel_requirements.GlmHybridKernelRequirements,
) -> Int {
  for ordinal, requirement in requirements.requirements() {
    if requirement.operation() is DsaQueryARmsNorm(_, _) {
      return ordinal
    }
  }
  abort("missing DSA q-a RMSNorm")
}

///|
fn find_query_b_ordinal(
  requirements : @glm_hybrid_kernel_requirements.GlmHybridKernelRequirements,
) -> Int {
  for ordinal, requirement in requirements.requirements() {
    if requirement.operation() is DsaQueryBProjection(_, _, _) {
      return ordinal
    }
  }
  abort("missing DSA q-b projection")
}

///|
fn find_key_value_b_ordinal(
  requirements : @glm_hybrid_kernel_requirements.GlmHybridKernelRequirements,
) -> Int {
  for ordinal, requirement in requirements.requirements() {
    if requirement.operation() is KeyValueLowRankProjection(_, _, _, _) {
      return ordinal
    }
  }
  abort("missing DSA kv-b projection")
}

///|
fn find_kv_split_norm_ordinal(
  requirements : @glm_hybrid_kernel_requirements.GlmHybridKernelRequirements,
) -> Int {
  for ordinal, requirement in requirements.requirements() {
    if requirement.operation() is QueryKeyRmsNorm(_) {
      return ordinal
    }
  }
  abort("missing DSA kv-a split RMSNorm")
}

///|
fn find_rotary_ordinal(
  requirements : @glm_hybrid_kernel_requirements.GlmHybridKernelRequirements,
) -> Int {
  for ordinal, requirement in requirements.requirements() {
    if requirement.operation() is GlmRotaryPosition(_, _) {
      return ordinal
    }
  }
  abort("missing GLM rotary requirement")
}

///|
fn find_dense_swiglu_ordinal(
  requirements : @glm_hybrid_kernel_requirements.GlmHybridKernelRequirements,
) -> Int {
  for ordinal, requirement in requirements.requirements() {
    if requirement.operation() is DenseSwiGlu(_, _) {
      return ordinal
    }
  }
  abort("missing dense SwiGLU requirement")
}

///|
fn find_kda_short_convolution_ordinal(
  requirements : @glm_hybrid_kernel_requirements.GlmHybridKernelRequirements,
) -> Int {
  for ordinal, requirement in requirements.requirements() {
    if requirement.operation() is KdaShortConvolution(_, _) {
      return ordinal
    }
  }
  abort("missing KDA short-convolution requirement")
}

///|
fn find_kda_projection_ordinal(
  requirements : @glm_hybrid_kernel_requirements.GlmHybridKernelRequirements,
) -> Int {
  for ordinal, requirement in requirements.requirements() {
    if requirement.operation() is KdaProjection(_, _) {
      return ordinal
    }
  }
  abort("missing KDA projection requirement")
}

///|
fn find_kda_output_gate_ordinal(
  requirements : @glm_hybrid_kernel_requirements.GlmHybridKernelRequirements,
) -> Int {
  for ordinal, requirement in requirements.requirements() {
    if requirement.operation() is KdaOutputGateProjection(_, _, _) {
      return ordinal
    }
  }
  abort("missing KDA output-gate requirement")
}

///|
fn find_kda_gated_norm_ordinal(
  requirements : @glm_hybrid_kernel_requirements.GlmHybridKernelRequirements,
) -> Int {
  for ordinal, requirement in requirements.requirements() {
    if requirement.operation() is KdaGatedRmsNorm(_, _, _) {
      return ordinal
    }
  }
  abort("missing KDA gated RMSNorm requirement")
}

///|
fn find_kda_output_projection_ordinal(
  requirements : @glm_hybrid_kernel_requirements.GlmHybridKernelRequirements,
) -> Int {
  for ordinal, requirement in requirements.requirements() {
    if requirement.operation() is KdaOutputProjection(_, _) {
      return ordinal
    }
  }
  abort("missing KDA output projection requirement")
}

///|
fn find_kda_decay_control_ordinal(
  requirements : @glm_hybrid_kernel_requirements.GlmHybridKernelRequirements,
) -> Int {
  for ordinal, requirement in requirements.requirements() {
    if requirement.operation() is KdaDecayControl(_, _, _) {
      return ordinal
    }
  }
  abort("missing KDA decay-control requirement")
}

///|
fn find_kda_recurrent_ordinal(
  requirements : @glm_hybrid_kernel_requirements.GlmHybridKernelRequirements,
) -> Int {
  for ordinal, requirement in requirements.requirements() {
    if requirement.operation() is KdaRecurrentUpdate(_, _, _) {
      return ordinal
    }
  }
  abort("missing KDA recurrent-update requirement")
}

///|
fn find_moe_router_ordinal(
  requirements : @glm_hybrid_kernel_requirements.GlmHybridKernelRequirements,
) -> Int {
  for ordinal, requirement in requirements.requirements() {
    if requirement.operation() is MoeRouterProjection(_, _, _) {
      return ordinal
    }
  }
  abort("missing MoE router projection requirement")
}

///|
fn find_moe_score_ordinal(
  requirements : @glm_hybrid_kernel_requirements.GlmHybridKernelRequirements,
) -> Int {
  for ordinal, requirement in requirements.requirements() {
    if requirement.operation() is MoeSigmoidAndCorrectionBias(_) {
      return ordinal
    }
  }
  abort("missing MoE sigmoid correction requirement")
}

///|
fn find_moe_grouped_routing_ordinal(
  requirements : @glm_hybrid_kernel_requirements.GlmHybridKernelRequirements,
) -> Int {
  for ordinal, requirement in requirements.requirements() {
    if requirement.operation()
      is MoeGroupedNoAuxiliaryLossTopK(_, _, _, _, _, _, _) {
      return ordinal
    }
  }
  abort("missing MoE grouped-routing requirement")
}

///|
fn find_named_tensor(
  manifest : @glm53_weights.Glm53TensorManifest,
  name : String,
) -> @glm53_weights.Glm53TensorDescriptor {
  for ordinal in 0..<manifest.tensor_count() {
    guard manifest.tensor(ordinal) is Some(tensor) else { continue }
    if tensor.name() == name {
      return tensor
    }
  }
  abort("missing named tensor")
}

///|
fn find_head_tensor(
  manifest : @glm53_weights.Glm53TensorManifest,
) -> @glm53_weights.Glm53TensorDescriptor {
  for ordinal in 0..<manifest.tensor_count() {
    guard manifest.tensor(ordinal) is Some(tensor) else { continue }
    if tensor.name() == "lm_head.weight" {
      return tensor
    }
  }
  abort("missing lm_head.weight")
}

///|
fn count_attention_output_tensors(
  manifest : @glm53_weights.Glm53TensorManifest,
  shape : ReadOnlyArray[Int64],
) -> Int {
  let mut count = 0
  for ordinal in 0..<manifest.tensor_count() {
    guard manifest.tensor(ordinal) is Some(tensor) else { continue }
    if tensor.dtype() == BFloat16 &&
      tensor.name().has_suffix(".self_attn.o_proj.weight") &&
      tensor.shape() == shape {
      count += 1
    }
  }
  count
}

///|
fn count_exact_pool_ape_tensors(
  manifest : @glm53_weights.Glm53TensorManifest,
) -> Int {
  let mut count = 0
  for ordinal in 0..<manifest.tensor_count() {
    guard manifest.tensor(ordinal) is Some(tensor) else { continue }
    if tensor.dtype() == BFloat16 &&
      tensor.shape() == [4L, 128L] &&
      tensor.name().has_suffix(".indexer.index_kpool_compress_ape") {
      count += 1
    }
  }
  count
}

///|
fn count_numeric_dsa_pooling(
  plan : @glm53_numeric_plan.Glm53NumericPlan,
) -> Int {
  let mut count = 0
  for ordinal in 0..<plan.operation_count() {
    guard plan.operation(ordinal) is Some(operation) else { continue }
    if operation.kind() == DsaIndexPooling {
      count += 1
    }
  }
  count
}

///|
fn count_numeric_dsa_low_rank_projection(
  plan : @glm53_numeric_plan.Glm53NumericPlan,
) -> Int {
  let mut count = 0
  for ordinal in 0..<plan.operation_count() {
    guard plan.operation(ordinal) is Some(operation) else { continue }
    if operation.kind() == DsaLowRankProjection {
      count += 1
    }
  }
  count
}
