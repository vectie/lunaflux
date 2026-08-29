///|
fn compile_candidate_sha256(bytes : BytesView) -> String {
  @crypto.bytes_to_hex_string(@crypto.sha256(bytes))
}

///|
fn compile_candidate_target(
  name : String,
) -> @catalog.DeviceTarget raise Fp8V2CompileCandidateError {
  let (major, minor) = match name {
    "sm_89" => (8, 9)
    "sm_90" => (9, 0)
    "sm_120" => (12, 0)
    _ => raise InvalidArguments
  }
  @catalog.DeviceTarget::new(
    compute_major=major,
    compute_minor=minor,
    supports_bf16=true,
    supports_cublas_lt=true,
  ) catch {
    _ => raise CandidateAdmission
  }
}

///|
fn compile_candidate_plan() -> @plan.ModelPlan raise Fp8V2CompileCandidateError {
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
  ) catch {
    _ => raise CandidateAdmission
  }
  @llama.build_paged_finite_e4m3_fp8_w8a8_v2(
    metadata,
    @llama.LlamaFp8W8A8V2Policy::untied_v2(),
  ) catch {
    _ => raise CandidateAdmission
  }
}

///|
fn compile_candidate_profile() -> @launch_contract.PagedKernelProfile raise Fp8V2CompileCandidateError {
  @launch_contract.PagedKernelProfile::new(
    id=@launch_contract.KernelProfileId::from_stable_int(1),
    max_query_rows=2,
    max_query_tokens=8,
    max_page_table_entries=16,
  ) catch {
    _ => raise CandidateAdmission
  }
}

///|
fn compile_candidate_policy(
  toolchain_sha256 : String,
) -> @luna_specialization.LunaCompilerPolicy raise Fp8V2CompileCandidateError {
  @luna_specialization.LunaCompilerPolicy::new(
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
  ) catch {
    _ => raise InvalidArguments
  }
}

///|
fn lower_compile_candidate(
  plan : @plan.ModelPlan,
  operation : Int,
  profile : @launch_contract.PagedKernelProfile,
  target : @catalog.DeviceTarget,
  compiler : @luna_specialization.LunaCompilerPolicy,
) -> @luna_cuda_fp8_projection_aot.Fp8ProjectionCudaLoweringV2 raise Fp8V2CompileCandidateError {
  let id = @plan.OperationId::new(operation) catch {
    _ => raise CandidateAdmission
  }
  let operands = @luna_cuda_fp8_projection_aot.derive_fp8_projection_operands_v2(
    plan,
    operation=id,
    profile,
  ) catch {
    _ => raise CandidateAdmission
  }
  @luna_cuda_fp8_projection_aot.lower_fp8_projection_cuda_aot_candidate_v2(
    plan,
    operation=id,
    profile,
    target,
    compiler,
    operands,
    @luna_cuda_fp8_projection_aot.Fp8ProjectionAotLimits::new(
      max_source_bytes=1048576,
      max_recipe_bytes=65536,
    ),
  ) catch {
    _ => raise CandidateAdmission
  }
}

///|
fn compile_candidate_manifest(
  target_name : String,
  plan : @plan.ModelPlan,
  profile : @launch_contract.PagedKernelProfile,
  compiler : @luna_specialization.LunaCompilerPolicy,
  simple : @luna_cuda_fp8_projection_aot.Fp8ProjectionCudaLoweringV2,
  gated : @luna_cuda_fp8_projection_aot.Fp8ProjectionCudaLoweringV2,
) -> Bytes {
  let identity = plan.identity()
  @utf8.encode(
    "schema=lunaflux-fp8-v2-compile-candidates.v1\n" +
    "model_content_sha256=" +
    identity.content().as_hex() +
    "\nmodel_plan_sha256=" +
    identity.plan().as_hex() +
    "\nprofile_id=" +
    profile.id().as_int().to_string() +
    "\nmax_query_rows=" +
    profile.max_query_rows().to_string() +
    "\nmax_query_tokens=" +
    profile.max_query_tokens().to_string() +
    "\nmax_page_table_entries=" +
    profile.max_page_table_entries().to_string() +
    "\ntoolchain_sha256=" +
    compiler.identity().toolchain_digest_sha256() +
    "\ncompiler_version=13.1.115\ntarget=" +
    target_name +
    "\ncandidate_count=2\n" +
    "simple_family=qkv_projection\nsimple_operation_id=2\n" +
    "simple_function_symbol=" +
    simple.function_symbol() +
    "\nsimple_source_sha256=" +
    simple.source_digest().as_hex() +
    "\nsimple_recipe_sha256=" +
    simple.recipe_digest().as_hex() +
    "\nsimple_source=simple.cu\nsimple_recipe=simple.recipe\n" +
    "gated_family=gated_mlp\ngated_operation_id=8\n" +
    "gated_function_symbol=" +
    gated.function_symbol() +
    "\ngated_source_sha256=" +
    gated.source_digest().as_hex() +
    "\ngated_recipe_sha256=" +
    gated.recipe_digest().as_hex() +
    "\ngated_source=gated.cu\ngated_recipe=gated.recipe\n" +
    "generated_lowerings=1\ncaller_source_authority=0\n" +
    "compiler_invoked=0\ndevice_opened=0\nruntime_authority=0\n" +
    "numerical_execution=0\nreadiness=0\n",
  )
}

///|
fn prepare_compile_candidates(
  target_name : String,
  toolchain_sha256 : String,
) -> Fp8V2CompileCandidateSet raise Fp8V2CompileCandidateError {
  let plan = compile_candidate_plan()
  let profile = compile_candidate_profile()
  let target = compile_candidate_target(target_name)
  let compiler = compile_candidate_policy(toolchain_sha256)
  let simple = lower_compile_candidate(plan, 2, profile, target, compiler)
  let gated = lower_compile_candidate(plan, 8, profile, target, compiler)
  let manifest = compile_candidate_manifest(
    target_name, plan, profile, compiler, simple, gated,
  )
  {
    target: target_name,
    model_content_sha256: plan.identity().content().as_hex(),
    model_plan_sha256: plan.identity().plan().as_hex(),
    simple,
    gated,
    manifest,
    manifest_sha256: compile_candidate_sha256(manifest),
  }
}
