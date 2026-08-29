///|
struct Fp8V2ProbeCandidates {
  plan : @plan.ModelPlan
  simple : @luna_cuda_fp8_projection_aot.Fp8ProjectionCudaLoweringV2
  gated : @luna_cuda_fp8_projection_aot.Fp8ProjectionCudaLoweringV2
}

///|
fn fp8_probe_plan() -> @plan.ModelPlan raise {
  let metadata = @spec.LlamaModelMetadata::from_verified_content(
    @spec.LlamaModelSpec::new(
      vocabulary_size=32,
      hidden_size=8,
      intermediate_size=16,
      layer_count=1,
      attention_head_count=4,
      kv_head_count=2,
      context_length=32,
      rope_theta=10000.0,
      weight_dtype=BFloat16,
      kv_dtype=BFloat16,
    ),
    @spec.ContentDigest::from_sha256(
      "234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef1",
    ),
  )
  @llama.build_paged_finite_e4m3_fp8_w8a8_v2(
    metadata,
    @llama.LlamaFp8W8A8V2Policy::untied_v2(),
  )
}

///|
fn fp8_probe_candidates(
  toolchain_sha256 : String,
) -> Fp8V2ProbeCandidates raise {
  require_fp8_probe(
    toolchain_sha256.length() == 64 &&
    toolchain_sha256.all(character => {
      (character >= '0' && character <= '9') ||
      (character >= 'a' && character <= 'f')
    }),
    "toolchain digest is not canonical SHA-256",
  )
  let plan = fp8_probe_plan()
  let profile = @launch_contract.PagedKernelProfile::new(
    id=@launch_contract.KernelProfileId::from_stable_int(1),
    max_query_rows=2,
    max_query_tokens=8,
    max_page_table_entries=16,
  )
  let target = @catalog.DeviceTarget::new(
    compute_major=12,
    compute_minor=0,
    supports_bf16=true,
    supports_cublas_lt=true,
  )
  let compiler = @luna_specialization.LunaCompilerPolicy::new(
    identity=@luna_specialization.LunaCompilerIdentity::new(
      toolchain_digest_sha256=toolchain_sha256,
      major=13,
      minor=1,
      patch=115,
    ),
    optimization_level=3,
    fused_multiply_add=false,
    reassociate=false,
    max_registers=128,
  )
  let limits = @luna_cuda_fp8_projection_aot.Fp8ProjectionAotLimits::new(
    max_source_bytes=1048576,
    max_recipe_bytes=65536,
  )
  fn lower(
    plan : @plan.ModelPlan,
    operation : Int,
    profile : @launch_contract.PagedKernelProfile,
    target : @catalog.DeviceTarget,
    compiler : @luna_specialization.LunaCompilerPolicy,
    limits : @luna_cuda_fp8_projection_aot.Fp8ProjectionAotLimits,
  ) -> @luna_cuda_fp8_projection_aot.Fp8ProjectionCudaLoweringV2 raise {
    let id = @plan.OperationId::new(operation)
    let operands = @luna_cuda_fp8_projection_aot.derive_fp8_projection_operands_v2(
      plan,
      operation=id,
      profile,
    )
    @luna_cuda_fp8_projection_aot.lower_fp8_projection_cuda_aot_candidate_v2(
      plan,
      operation=id,
      profile,
      target,
      compiler,
      operands,
      limits,
    )
  }
  {
    plan,
    simple: lower(plan, 2, profile, target, compiler, limits),
    gated: lower(plan, 8, profile, target, compiler, limits),
  }
}
