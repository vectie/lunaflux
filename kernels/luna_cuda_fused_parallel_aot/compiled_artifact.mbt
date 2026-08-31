///|
fn fused_compiled_family_name(family : FusedParallelFamily) -> String {
  match family {
    QkvPositionedRopePagedKvWrite => "qkv-positioned-rope-paged-kvwrite"
    ResidualRmsNorm => "residual-rmsnorm"
  }
}

///|
fn fused_compiled_target_name(target : @catalog.DeviceTarget) -> String {
  "sm_" +
  target.compute_major().to_string() +
  target.compute_minor().to_string()
}

///|
fn fused_receipt_line_value(
  lines : ArrayView[String],
  index : Int,
  key : String,
) -> String raise FusedParallelAotError {
  if index < 0 || index >= lines.length() {
    raise FusedParallelInvalid(CompileReceipt)
  }
  let prefix = key + "="
  let line = lines[index]
  if !line.has_prefix(prefix) || line.length() == prefix.length() {
    raise FusedParallelInvalid(CompileReceipt)
  }
  line[prefix.length():].to_owned()
}

///|
fn fused_require_recipe_abi(
  recipe : Bytes,
  abi : FusedParallelKernelAbi,
  dispatch_canary_per_token : UInt,
  qualification_candidate : FusedParallelDigest?,
) -> Unit raise FusedParallelAotError {
  let document = @utf8.decode(recipe) catch {
    _ => raise FusedParallelInvalid(Recipe)
  }
  let expected = "abi=" + abi.canonical_name()
  let mut abi_count = 0
  let mut canary_count = 0
  let mut diagnostic_absent_count = 0
  let mut qualification_count = 0
  for line in document.split("\n") {
    if line.has_prefix("abi=") {
      abi_count += 1
      if line != expected {
        raise FusedParallelMismatch(KernelAbi)
      }
    }
    if line.has_prefix("dispatch_canary=") {
      canary_count += 1
      if line != "dispatch_canary=" + dispatch_canary_per_token.to_string() {
        raise FusedParallelMismatch(KernelAbi)
      }
    }
    if line.has_prefix("diagnostic_canary=") {
      diagnostic_absent_count += 1
      if line != "diagnostic_canary=absent" {
        raise FusedParallelMismatch(KernelAbi)
      }
    }
    if line.has_prefix("qualification_candidate_sha256=") {
      qualification_count += 1
      guard qualification_candidate is Some(parent) &&
        line == "qualification_candidate_sha256=" + parent.as_hex() else {
        raise FusedParallelMismatch(Digest)
      }
    }
  }
  match abi {
    ResidualRmsNormProductionAbiV2
    | QkvPositionedRopePagedKvWriteProductionAbiV2
    | QwenQkvQkNormPositionedRopePagedKvWriteProductionAbiV4 =>
      if abi_count != 1 ||
        canary_count != 0 ||
        diagnostic_absent_count != 1 ||
        qualification_count != 1 ||
        dispatch_canary_per_token != 0U ||
        qualification_candidate is None {
        raise FusedParallelMismatch(KernelAbi)
      }
    _ =>
      if abi_count != 1 ||
        canary_count != 1 ||
        diagnostic_absent_count != 0 ||
        qualification_count != 0 ||
        dispatch_canary_per_token == 0U ||
        qualification_candidate is Some(_) {
        raise FusedParallelMismatch(KernelAbi)
      }
  }
}

///|
fn fused_compiled_binding_record(
  family : FusedParallelFamily,
  candidate_digest : FusedParallelDigest,
  source_digest : FusedParallelDigest,
  recipe_digest : FusedParallelDigest,
  compiler : @luna_specialization.LunaCompilerPolicy,
  driver_identity_sha256 : String,
  target : @catalog.DeviceTarget,
  dimensions : @launch_contract.AotLaunchDimensions,
  function_symbol : String,
  abi : FusedParallelKernelAbi,
  dispatch_canary_per_token : UInt,
  qualification_candidate : FusedParallelDigest?,
  module_digest : FusedParallelDigest,
  receipt_digest : FusedParallelDigest,
) -> Bytes {
  let identity = compiler.identity()
  let qualification_record = match qualification_candidate {
    Some(parent) => "qualification_candidate_sha256=" + parent.as_hex() + "\n"
    None => "qualification_candidate_sha256=absent\n"
  }
  @utf8.encode(
    "schema=lunaflux-fused-parallel-compiled-candidate-binding.v1\n" +
    "family=" +
    fused_compiled_family_name(family) +
    "\n" +
    "candidate_sha256=" +
    candidate_digest.as_hex() +
    "\n" +
    "source_sha256=" +
    source_digest.as_hex() +
    "\n" +
    "recipe_sha256=" +
    recipe_digest.as_hex() +
    "\n" +
    "toolchain_sha256=" +
    identity.toolchain_digest_sha256() +
    "\n" +
    "driver_identity_sha256=" +
    driver_identity_sha256 +
    "\n" +
    "compiler_version=" +
    identity.major().to_string() +
    "." +
    identity.minor().to_string() +
    "." +
    identity.patch().to_string() +
    "\n" +
    "optimization=" +
    compiler.optimization_level().to_string() +
    "\n" +
    "fmad=" +
    compiler.fused_multiply_add().to_string() +
    "\n" +
    "reassociate=" +
    compiler.reassociate().to_string() +
    "\n" +
    "max_registers=" +
    compiler.max_registers().to_string() +
    "\n" +
    "target=" +
    fused_compiled_target_name(target) +
    "\n" +
    "grid=" +
    dimensions.grid_x().to_string() +
    "," +
    dimensions.grid_y().to_string() +
    "," +
    dimensions.grid_z().to_string() +
    "\n" +
    "block=" +
    dimensions.block_x().to_string() +
    "," +
    dimensions.block_y().to_string() +
    "," +
    dimensions.block_z().to_string() +
    "\n" +
    "shared_memory_bytes=" +
    dimensions.shared_memory_bytes().to_string() +
    "\n" +
    "function_symbol=" +
    function_symbol +
    "\n" +
    "abi=" +
    abi.canonical_name() +
    "\n" +
    "dispatch_canary_per_token=" +
    dispatch_canary_per_token.to_string() +
    "\n" +
    qualification_record +
    "cubin_sha256=" +
    module_digest.as_hex() +
    "\n" +
    "compile_receipt_sha256=" +
    receipt_digest.as_hex() +
    "\n" +
    "deterministic=1\n" +
    "candidate_only=true\n" +
    "compiler_authority=absent\n" +
    "deployment_approval=absent\n" +
    "promotion_authority=absent\n" +
    "manifest_bindable=false\n",
  )
}

///|
fn fused_bind_compiled_candidate(
  family : FusedParallelFamily,
  candidate_digest : FusedParallelDigest,
  source : Bytes,
  source_digest : FusedParallelDigest,
  recipe : Bytes,
  recipe_digest : FusedParallelDigest,
  compiler : @luna_specialization.LunaCompilerPolicy,
  target : @catalog.DeviceTarget,
  dimensions : @launch_contract.AotLaunchDimensions,
  function_symbol : String,
  abi : FusedParallelKernelAbi,
  dispatch_canary_per_token : UInt,
  qualification_candidate : FusedParallelDigest?,
  first_module : Bytes,
  second_module : Bytes,
  compile_receipt : Bytes,
  expected_compile_receipt_digest : FusedParallelDigest,
  limits : FusedParallelCompiledArtifactLimits,
) -> FusedParallelCompiledArtifactBinding raise FusedParallelAotError {
  if first_module.length() == 0 || second_module.length() == 0 {
    raise FusedParallelInvalid(CompiledArtifact)
  }
  match abi {
    ResidualRmsNormProductionAbiV2
    | QkvPositionedRopePagedKvWriteProductionAbiV2
    | QwenQkvQkNormPositionedRopePagedKvWriteProductionAbiV4 =>
      if dispatch_canary_per_token != 0U || qualification_candidate is None {
        raise FusedParallelInvalid(CompiledArtifact)
      }
    _ =>
      if dispatch_canary_per_token == 0U || qualification_candidate is Some(_) {
        raise FusedParallelInvalid(CompiledArtifact)
      }
  }
  let purpose = match abi {
    ResidualRmsNormProductionAbiV2
    | QkvPositionedRopePagedKvWriteProductionAbiV2
    | QwenQkvQkNormPositionedRopePagedKvWriteProductionAbiV4 =>
      ProductionFastPath
    _ => QualificationDiagnostics
  }
  if first_module.length().to_int64() > limits.maximum_module_bytes ||
    second_module.length().to_int64() > limits.maximum_module_bytes {
    raise FusedParallelLimitExceeded(CompiledArtifact)
  }
  if compile_receipt.length() == 0 ||
    compile_receipt.length() > limits.maximum_receipt_bytes {
    raise FusedParallelLimitExceeded(CompileReceipt)
  }
  let compile_receipt_digest = fused_parallel_digest(compile_receipt)
  if compile_receipt_digest != expected_compile_receipt_digest {
    raise FusedParallelMismatch(CompileReceipt)
  }
  if fused_parallel_digest(source) != source_digest ||
    fused_parallel_digest(recipe) != recipe_digest {
    raise FusedParallelMismatch(Digest)
  }
  fused_require_recipe_abi(
    recipe, abi, dispatch_canary_per_token, qualification_candidate,
  )
  let first_digest = fused_parallel_digest(first_module)
  let second_digest = fused_parallel_digest(second_module)
  if first_digest != second_digest || first_module != second_module {
    raise FusedParallelMismatch(Determinism)
  }
  let receipt_document = @utf8.decode(compile_receipt) catch {
    _ => raise FusedParallelInvalid(CompileReceipt)
  }
  let receipt_lines = receipt_document
    .split("\n")
    .map(line => line.to_owned())
    .to_array()
  if receipt_lines.length() != 12 || receipt_lines[11] != "" {
    raise FusedParallelInvalid(CompileReceipt)
  }
  let driver_identity_sha256 = fused_receipt_line_value(
    receipt_lines, 4, "driver_identity_sha256",
  )
  if !fused_parallel_is_sha256(driver_identity_sha256) {
    raise FusedParallelInvalid(CompileReceipt)
  }
  let expected_receipt = @utf8.encode(
    "schema=lunaflux-luna-cuda-aot-evidence-v1\n" +
    "source_sha256=" +
    source_digest.as_hex() +
    "\n" +
    "recipe_sha256=" +
    recipe_digest.as_hex() +
    "\n" +
    "toolchain_sha256=" +
    compiler.identity().toolchain_digest_sha256() +
    "\n" +
    "driver_identity_sha256=" +
    driver_identity_sha256 +
    "\n" +
    "target=" +
    fused_compiled_target_name(target) +
    "\n" +
    "function_symbol=" +
    function_symbol +
    "\n" +
    "artifact_sha256=" +
    first_digest.as_hex() +
    "\n" +
    "first_build_sha256=" +
    first_digest.as_hex() +
    "\n" +
    "second_build_sha256=" +
    second_digest.as_hex() +
    "\n" +
    "deterministic=1\n",
  )
  if compile_receipt != expected_receipt {
    raise FusedParallelMismatch(CompileReceipt)
  }
  let binding_record = fused_compiled_binding_record(
    family, candidate_digest, source_digest, recipe_digest, compiler, driver_identity_sha256,
    target, dimensions, function_symbol, abi, dispatch_canary_per_token, qualification_candidate,
    first_digest, compile_receipt_digest,
  )
  if binding_record.length() > limits.maximum_binding_bytes {
    raise FusedParallelLimitExceeded(Evidence)
  }
  {
    family,
    candidate_digest,
    source_digest,
    recipe_digest,
    toolchain_digest_sha256: compiler.identity().toolchain_digest_sha256(),
    driver_identity_sha256,
    target,
    launch_dimensions: dimensions,
    function_symbol,
    kernel_abi: abi,
    purpose,
    dispatch_canary_per_token,
    qualification_candidate,
    module_bytes: first_module,
    module_digest: first_digest,
    compile_receipt,
    compile_receipt_digest,
    binding_record,
    binding_digest: fused_parallel_digest(binding_record),
  }
}

///|
/// Binds deterministic offline compiler output to one exact fused QKV/RoPE/KV
/// candidate without granting runtime or release authority.
pub fn bind_fused_qkv_rope_kv_compiled_candidate(
  candidate : FusedQkvRopeKvWriteCudaCandidate,
  first_module : Bytes,
  second_module : Bytes,
  compile_receipt : Bytes,
  expected_compile_receipt_digest : FusedParallelDigest,
  limits : FusedParallelCompiledArtifactLimits,
) -> FusedParallelCompiledArtifactBinding raise FusedParallelAotError {
  fused_bind_compiled_candidate(
    QkvPositionedRopePagedKvWrite,
    candidate.candidate_digest,
    candidate.source,
    candidate.source_digest,
    candidate.recipe,
    candidate.recipe_digest,
    candidate.compiler,
    candidate.target,
    candidate.dimensions,
    candidate.function_symbol,
    QkvPositionedRopePagedKvWriteAbiV1,
    candidate.dispatch_canary,
    None,
    first_module,
    second_module,
    compile_receipt,
    expected_compile_receipt_digest,
    limits,
  )
}

///|
/// Binds deterministic offline compiler output to one exact fused
/// residual/RMSNorm candidate without granting runtime or release authority.
pub fn bind_fused_residual_rmsnorm_compiled_candidate(
  candidate : FusedResidualRmsNormCudaCandidate,
  first_module : Bytes,
  second_module : Bytes,
  compile_receipt : Bytes,
  expected_compile_receipt_digest : FusedParallelDigest,
  limits : FusedParallelCompiledArtifactLimits,
) -> FusedParallelCompiledArtifactBinding raise FusedParallelAotError {
  fused_bind_compiled_candidate(
    ResidualRmsNorm,
    candidate.candidate_digest,
    candidate.source,
    candidate.source_digest,
    candidate.recipe,
    candidate.recipe_digest,
    candidate.compiler,
    candidate.target,
    candidate.dimensions,
    candidate.function_symbol,
    ResidualRmsNormAbiV1,
    candidate.dispatch_canary,
    None,
    first_module,
    second_module,
    compile_receipt,
    expected_compile_receipt_digest,
    limits,
  )
}

///|
/// Binds deterministic compiler output for the canary-free production ABI.
/// The result remains non-manifest-bindable and retains the exact
/// qualification candidate it was derived from.
pub fn bind_fused_residual_rmsnorm_production_compiled_candidate(
  candidate : FusedResidualRmsNormProductionCudaCandidate,
  first_module : Bytes,
  second_module : Bytes,
  compile_receipt : Bytes,
  expected_compile_receipt_digest : FusedParallelDigest,
  limits : FusedParallelCompiledArtifactLimits,
) -> FusedParallelCompiledArtifactBinding raise FusedParallelAotError {
  let qualification = candidate.qualification
  fused_bind_compiled_candidate(
    ResidualRmsNorm,
    candidate.candidate_digest,
    candidate.source,
    candidate.source_digest,
    candidate.recipe,
    candidate.recipe_digest,
    qualification.compiler,
    qualification.target,
    qualification.dimensions,
    candidate.function_symbol,
    ResidualRmsNormProductionAbiV2,
    0U,
    Some(qualification.candidate_digest),
    first_module,
    second_module,
    compile_receipt,
    expected_compile_receipt_digest,
    limits,
  )
}

///|
/// Binds deterministic compiler output for the canary-free production QKV
/// composite. The binding remains inert pending external approval.
pub fn bind_fused_qkv_rope_paged_kv_write_production_compiled_candidate(
  candidate : FusedQkvRopeKvWriteProductionCudaCandidate,
  first_module : Bytes,
  second_module : Bytes,
  compile_receipt : Bytes,
  expected_compile_receipt_digest : FusedParallelDigest,
  limits : FusedParallelCompiledArtifactLimits,
) -> FusedParallelCompiledArtifactBinding raise FusedParallelAotError {
  let qualification = candidate.qualification
  fused_bind_compiled_candidate(
    QkvPositionedRopePagedKvWrite,
    candidate.candidate_digest,
    candidate.source,
    candidate.source_digest,
    candidate.recipe,
    candidate.recipe_digest,
    qualification.compiler,
    qualification.target,
    qualification.dimensions,
    candidate.function_symbol,
    QkvPositionedRopePagedKvWriteProductionAbiV2,
    0U,
    Some(qualification.candidate_digest),
    first_module,
    second_module,
    compile_receipt,
    expected_compile_receipt_digest,
    limits,
  )
}

///|
/// Binds deterministic offline output for the Qwen qualification ABI.
pub fn bind_fused_qwen_qkv_qknorm_rope_kv_compiled_candidate(
  candidate : FusedQwenQkvQkNormRopeKvWriteCudaCandidate,
  first_module : Bytes,
  second_module : Bytes,
  compile_receipt : Bytes,
  expected_compile_receipt_digest : FusedParallelDigest,
  limits : FusedParallelCompiledArtifactLimits,
) -> FusedParallelCompiledArtifactBinding raise FusedParallelAotError {
  fused_bind_compiled_candidate(
    QkvPositionedRopePagedKvWrite,
    candidate.candidate_digest,
    candidate.source,
    candidate.source_digest,
    candidate.recipe,
    candidate.recipe_digest,
    candidate.compiler,
    candidate.target,
    candidate.dimensions,
    candidate.function_symbol,
    QwenQkvQkNormPositionedRopePagedKvWriteAbiV3,
    candidate.dispatch_canary,
    None,
    first_module,
    second_module,
    compile_receipt,
    expected_compile_receipt_digest,
    limits,
  )
}

///|
/// Binds deterministic offline output for the canary-free Qwen ABI.
pub fn bind_fused_qwen_qkv_qknorm_rope_kv_production_compiled_candidate(
  candidate : FusedQwenQkvQkNormRopeKvWriteProductionCudaCandidate,
  first_module : Bytes,
  second_module : Bytes,
  compile_receipt : Bytes,
  expected_compile_receipt_digest : FusedParallelDigest,
  limits : FusedParallelCompiledArtifactLimits,
) -> FusedParallelCompiledArtifactBinding raise FusedParallelAotError {
  let qualification = candidate.qualification
  fused_bind_compiled_candidate(
    QkvPositionedRopePagedKvWrite,
    candidate.candidate_digest,
    candidate.source,
    candidate.source_digest,
    candidate.recipe,
    candidate.recipe_digest,
    qualification.compiler,
    qualification.target,
    qualification.dimensions,
    candidate.function_symbol,
    QwenQkvQkNormPositionedRopePagedKvWriteProductionAbiV4,
    0U,
    Some(qualification.candidate_digest),
    first_module,
    second_module,
    compile_receipt,
    expected_compile_receipt_digest,
    limits,
  )
}
