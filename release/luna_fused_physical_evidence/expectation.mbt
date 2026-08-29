///|
fn fused_physical_validate_compiled_binding(
  family : @luna_cuda_fused_parallel_aot.FusedParallelFamily,
  candidate_digest : @luna_cuda_fused_parallel_aot.FusedParallelDigest,
  source_digest : @luna_cuda_fused_parallel_aot.FusedParallelDigest,
  recipe_digest : @luna_cuda_fused_parallel_aot.FusedParallelDigest,
  compiler : @luna_specialization.LunaCompilerPolicy,
  target : @catalog.DeviceTarget,
  dimensions : @launch_contract.AotLaunchDimensions,
  function_symbol : String,
  abi : @luna_cuda_fused_parallel_aot.FusedParallelKernelAbi,
  binding : @luna_cuda_fused_parallel_aot.FusedParallelCompiledArtifactBinding,
) -> Unit raise FusedPhysicalEvidenceError {
  if binding.family() != family ||
    binding.candidate_digest() != candidate_digest ||
    binding.source_digest() != source_digest ||
    binding.recipe_digest() != recipe_digest {
    raise FusedPhysicalEvidenceMismatch(Candidate)
  }
  if binding.toolchain_digest_sha256() !=
    compiler.identity().toolchain_digest_sha256() {
    raise FusedPhysicalEvidenceMismatch(Compiler)
  }
  if binding.target() != target ||
    binding.launch_dimensions() != dimensions ||
    binding.function_symbol() != function_symbol ||
    binding.kernel_abi() != abi {
    raise FusedPhysicalEvidenceMismatch(Artifact)
  }
  if binding.manifest_bindable() {
    raise FusedPhysicalEvidenceMismatch(Authority)
  }
  ignore(fused_physical_nonzero_digest(binding.driver_identity_sha256()))
}

///|
fn fused_physical_expected_qkv(
  candidate : @luna_cuda_fused_parallel_aot.FusedQkvRopeKvWriteCudaCandidate,
  binding : @luna_cuda_fused_parallel_aot.FusedParallelCompiledArtifactBinding,
) -> ExpectedFusedPhysicalFamily raise FusedPhysicalEvidenceError {
  let family = @luna_cuda_fused_parallel_aot.QkvPositionedRopePagedKvWrite
  let abi = @luna_cuda_fused_parallel_aot.QkvPositionedRopePagedKvWriteAbiV1
  let policy = candidate.numerical_policy()
  if candidate.manifest_bindable() ||
    policy != QkvRopeKvWriteOrderedF32TranscendentalToleranceV1 {
    raise FusedPhysicalEvidenceMismatch(Candidate)
  }
  fused_physical_validate_compiled_binding(
    family,
    candidate.candidate_digest(),
    candidate.source_digest(),
    candidate.recipe_digest(),
    candidate.compiler_policy(),
    candidate.target(),
    candidate.launch_dimensions(),
    candidate.function_symbol(),
    abi,
    binding,
  )
  let qkv_fallback = candidate.qkv_fallback()
  let rope_fallback = candidate.rope_fallback()
  let kv_write_fallback = candidate.kv_write_fallback()
  {
    family_name: "qkv-positioned-rope-paged-kvwrite",
    candidate_digest: candidate.candidate_digest(),
    source_digest: candidate.source_digest(),
    recipe_digest: candidate.recipe_digest(),
    recipe_bytes: candidate.recipe_bytes(),
    compiler: candidate.compiler_policy(),
    target: candidate.target(),
    profile: candidate.profile(),
    dimensions: candidate.launch_dimensions(),
    function_symbol: candidate.function_symbol(),
    abi,
    module_digest: binding.module_digest(),
    receipt_digest: binding.compile_receipt_digest(),
    driver_identity_sha256: binding.driver_identity_sha256(),
    numerical_policy: policy,
    dispatch_canary: candidate.dispatch_canary(),
    fallbacks: ReadOnlyArray::from_array([
      {
        operation: qkv_fallback.operation_id().as_int(),
        source: fused_physical_nonzero_digest(
          qkv_fallback.source_digest().as_hex(),
        ),
        recipe: fused_physical_nonzero_digest(
          qkv_fallback.recipe_digest().as_hex(),
        ),
      },
      {
        operation: rope_fallback.operation_id().as_int(),
        source: fused_physical_nonzero_digest(
          rope_fallback.source_digest().as_hex(),
        ),
        recipe: fused_physical_nonzero_digest(
          rope_fallback.recipe_digest().as_hex(),
        ),
      },
      {
        operation: kv_write_fallback.operation_id().as_int(),
        source: fused_physical_nonzero_digest(
          kv_write_fallback.source_digest().as_hex(),
        ),
        recipe: fused_physical_nonzero_digest(
          kv_write_fallback.recipe_digest().as_hex(),
        ),
      },
    ]),
  }
}

///|
fn fused_physical_expected_residual(
  candidate : @luna_cuda_fused_parallel_aot.FusedResidualRmsNormCudaCandidate,
  binding : @luna_cuda_fused_parallel_aot.FusedParallelCompiledArtifactBinding,
) -> ExpectedFusedPhysicalFamily raise FusedPhysicalEvidenceError {
  let family = @luna_cuda_fused_parallel_aot.ResidualRmsNorm
  let abi = @luna_cuda_fused_parallel_aot.ResidualRmsNormAbiV1
  let policy = candidate.numerical_policy()
  if candidate.manifest_bindable() ||
    policy != ResidualRmsNormBlock128F32ReductionToleranceV1 {
    raise FusedPhysicalEvidenceMismatch(Candidate)
  }
  fused_physical_validate_compiled_binding(
    family,
    candidate.candidate_digest(),
    candidate.source_digest(),
    candidate.recipe_digest(),
    candidate.compiler_policy(),
    candidate.target(),
    candidate.launch_dimensions(),
    candidate.function_symbol(),
    abi,
    binding,
  )
  let residual_fallback = candidate.residual_fallback()
  let norm_fallback = candidate.norm_fallback()
  {
    family_name: "residual-rmsnorm",
    candidate_digest: candidate.candidate_digest(),
    source_digest: candidate.source_digest(),
    recipe_digest: candidate.recipe_digest(),
    recipe_bytes: candidate.recipe_bytes(),
    compiler: candidate.compiler_policy(),
    target: candidate.target(),
    profile: candidate.profile(),
    dimensions: candidate.launch_dimensions(),
    function_symbol: candidate.function_symbol(),
    abi,
    module_digest: binding.module_digest(),
    receipt_digest: binding.compile_receipt_digest(),
    driver_identity_sha256: binding.driver_identity_sha256(),
    numerical_policy: policy,
    dispatch_canary: candidate.dispatch_canary(),
    fallbacks: ReadOnlyArray::from_array([
      {
        operation: residual_fallback.operation_id().as_int(),
        source: fused_physical_nonzero_digest(
          residual_fallback.source_digest().as_hex(),
        ),
        recipe: fused_physical_nonzero_digest(
          residual_fallback.recipe_digest().as_hex(),
        ),
      },
      {
        operation: norm_fallback.operation_id().as_int(),
        source: fused_physical_nonzero_digest(
          norm_fallback.source_digest().as_hex(),
        ),
        recipe: fused_physical_nonzero_digest(
          norm_fallback.recipe_digest().as_hex(),
        ),
      },
    ]),
  }
}

///|
fn fused_physical_validate_expected_family(
  expected : ExpectedFusedPhysicalFamily,
) -> Unit raise FusedPhysicalEvidenceError {
  if expected.target.compute_major() != 12 ||
    expected.target.compute_minor() != 0 ||
    !expected.target.supports_bf16() {
    raise FusedPhysicalEvidenceMismatch(Target)
  }
  if expected.dimensions.block_x() != 128 ||
    expected.dimensions.block_y() != 1 ||
    expected.dimensions.block_z() != 1 ||
    expected.dimensions.grid_x() != expected.profile.max_query_tokens() ||
    expected.dimensions.grid_y() != 1 ||
    expected.dimensions.grid_z() != 1 {
    raise FusedPhysicalEvidenceMismatch(Geometry)
  }
}

///|
fn fused_physical_recipe_identity(
  recipe : Bytes,
  key : String,
) -> @luna_cuda_fused_parallel_aot.FusedParallelDigest raise FusedPhysicalEvidenceError {
  let document = @utf8.decode(recipe) catch {
    _ => raise FusedPhysicalEvidenceInvalid(Candidate)
  }
  let prefix = key + "="
  let mut found : @luna_cuda_fused_parallel_aot.FusedParallelDigest? = None
  for line in document.split("\n") {
    if line.has_prefix(prefix) {
      if found is Some(_) || line.length() == prefix.length() {
        raise FusedPhysicalEvidenceInvalid(Candidate)
      }
      found = Some(
        fused_physical_nonzero_digest(line[prefix.length():].to_owned()),
      )
    }
  }
  match found {
    Some(value) => value
    None => raise FusedPhysicalEvidenceInvalid(Candidate)
  }
}

///|
fn fused_physical_validate_expected_pair(
  qkv : ExpectedFusedPhysicalFamily,
  residual : ExpectedFusedPhysicalFamily,
) -> Unit raise FusedPhysicalEvidenceError {
  if qkv.compiler != residual.compiler ||
    qkv.target != residual.target ||
    qkv.profile != residual.profile {
    raise FusedPhysicalEvidenceMismatch(Candidate)
  }
  for key in ["model_content_sha256", "model_plan_sha256"] {
    if fused_physical_recipe_identity(qkv.recipe_bytes, key) !=
      fused_physical_recipe_identity(residual.recipe_bytes, key) {
      raise FusedPhysicalEvidenceMismatch(Candidate)
    }
  }
}
