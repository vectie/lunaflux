///|
fn fused_physical_require_family_digest(
  reader : FusedPhysicalReader,
  prefix : String,
  field : String,
  expected : @luna_cuda_fused_parallel_aot.FusedParallelDigest,
  rule : FusedPhysicalEvidenceRule,
) -> Unit raise FusedPhysicalEvidenceError {
  reader.require_digest(prefix + field, expected, rule)
}

///|
fn fused_physical_admit_family(
  reader : FusedPhysicalReader,
  prefix : String,
  expected : ExpectedFusedPhysicalFamily,
) -> AdmittedFusedPhysicalFamily raise FusedPhysicalEvidenceError {
  reader.require_literal(prefix + "family", expected.family_name, Family)
  fused_physical_require_family_digest(
    reader,
    prefix,
    "candidate_sha256",
    expected.candidate_digest,
    Candidate,
  )
  fused_physical_require_family_digest(
    reader,
    prefix,
    "source_sha256",
    expected.source_digest,
    Candidate,
  )
  fused_physical_require_family_digest(
    reader,
    prefix,
    "recipe_sha256",
    expected.recipe_digest,
    Candidate,
  )
  let identity = expected.compiler.identity()
  reader.require_literal(
    prefix + "toolchain_sha256",
    identity.toolchain_digest_sha256(),
    Compiler,
  )
  reader.require_literal(
    prefix + "driver_identity_sha256",
    expected.driver_identity_sha256,
    Compiler,
  )
  reader.require_int(prefix + "compiler_major", identity.major(), Compiler)
  reader.require_int(prefix + "compiler_minor", identity.minor(), Compiler)
  reader.require_int(prefix + "compiler_patch", identity.patch(), Compiler)
  reader.require_int(
    prefix + "optimization_level",
    expected.compiler.optimization_level(),
    Compiler,
  )
  reader.require_literal(
    prefix + "fmad",
    expected.compiler.fused_multiply_add().to_string(),
    Compiler,
  )
  reader.require_literal(
    prefix + "reassociate",
    expected.compiler.reassociate().to_string(),
    Compiler,
  )
  reader.require_int(
    prefix + "max_registers",
    expected.compiler.max_registers(),
    Compiler,
  )
  reader.require_literal(prefix + "target", "sm_120", Target)
  reader.require_literal(
    prefix + "target_supports_bf16",
    expected.target.supports_bf16().to_string(),
    Target,
  )
  reader.require_literal(
    prefix + "target_supports_cublas_lt",
    expected.target.supports_cublas_lt().to_string(),
    Target,
  )
  reader.require_int(
    prefix + "profile_id",
    expected.profile.id().as_int(),
    Geometry,
  )
  reader.require_int(
    prefix + "max_query_rows",
    expected.profile.max_query_rows(),
    Geometry,
  )
  reader.require_int(
    prefix + "max_query_tokens",
    expected.profile.max_query_tokens(),
    Geometry,
  )
  reader.require_int(
    prefix + "max_page_table_entries",
    expected.profile.max_page_table_entries(),
    Geometry,
  )
  reader.require_int(prefix + "grid_x", expected.dimensions.grid_x(), Geometry)
  reader.require_int(prefix + "grid_y", expected.dimensions.grid_y(), Geometry)
  reader.require_int(prefix + "grid_z", expected.dimensions.grid_z(), Geometry)
  reader.require_int(
    prefix + "block_x",
    expected.dimensions.block_x(),
    Geometry,
  )
  reader.require_int(
    prefix + "block_y",
    expected.dimensions.block_y(),
    Geometry,
  )
  reader.require_int(
    prefix + "block_z",
    expected.dimensions.block_z(),
    Geometry,
  )
  reader.require_int(
    prefix + "shared_memory_bytes",
    expected.dimensions.shared_memory_bytes(),
    Geometry,
  )
  reader.require_literal(
    prefix + "function_symbol",
    expected.function_symbol,
    Artifact,
  )
  reader.require_literal(
    prefix + "typed_abi",
    expected.abi.canonical_name(),
    Artifact,
  )
  fused_physical_require_family_digest(
    reader,
    prefix,
    "first_build_cubin_sha256",
    expected.module_digest,
    Artifact,
  )
  fused_physical_require_family_digest(
    reader,
    prefix,
    "second_build_cubin_sha256",
    expected.module_digest,
    Artifact,
  )
  fused_physical_require_family_digest(
    reader,
    prefix,
    "compile_receipt_sha256",
    expected.receipt_digest,
    Artifact,
  )
  reader.require_literal(prefix + "deterministic", "true", Artifact)
  reader.require_literal(
    prefix + "numerical_policy",
    expected.numerical_policy.name(),
    Numerical,
  )
  let maximum_absolute = expected.numerical_policy.maximum_absolute_error_ppb()
  let maximum_relative = expected.numerical_policy.maximum_relative_error_ppb()
  reader.require_int64(
    prefix + "maximum_absolute_error_ppb",
    maximum_absolute,
    Numerical,
  )
  reader.require_int64(
    prefix + "maximum_relative_error_ppb",
    maximum_relative,
    Numerical,
  )
  let observed_absolute = reader.read_int64(
    prefix + "observed_maximum_absolute_error_ppb",
  )
  let observed_relative = reader.read_int64(
    prefix + "observed_maximum_relative_error_ppb",
  )
  if observed_absolute < 0L ||
    observed_absolute > maximum_absolute ||
    observed_relative < 0L ||
    observed_relative > maximum_relative {
    raise FusedPhysicalEvidenceMismatch(Numerical)
  }
  reader.require_uint(
    prefix + "dispatch_canary_per_token",
    expected.dispatch_canary,
    DispatchCanary,
  )
  let aggregate_canary = expected.dispatch_canary *
    FUSED_PHYSICAL_REFEREE_TOKEN_COUNT
  reader.require_uint(
    prefix + "expected_dispatch_canary",
    aggregate_canary,
    DispatchCanary,
  )
  let observed_canary = reader.read_uint(prefix + "observed_dispatch_canary")
  if observed_canary != aggregate_canary {
    raise FusedPhysicalEvidenceMismatch(DispatchCanary)
  }
  reader.require_literal(
    prefix + "dispatch_canary_pass",
    "true",
    DispatchCanary,
  )
  reader.require_literal(prefix + "sanitizer_status", "pass", Sanitizer)
  reader.require_literal(prefix + "race_status", "pass", Sanitizer)
  reader.require_literal(prefix + "initcheck_status", "pass", Sanitizer)
  {
    candidate_digest: expected.candidate_digest,
    source_digest: expected.source_digest,
    recipe_digest: expected.recipe_digest,
    module_digest: expected.module_digest,
    receipt_digest: expected.receipt_digest,
    observed_absolute_error_ppb: observed_absolute,
    observed_relative_error_ppb: observed_relative,
    dispatch_canary: observed_canary,
    dispatch_canary_per_token: expected.dispatch_canary,
    fallbacks: expected.fallbacks,
  }
}
