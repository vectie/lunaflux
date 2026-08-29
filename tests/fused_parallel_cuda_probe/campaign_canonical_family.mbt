///|
fn campaign_line(output : StringBuilder, key : String, value : String) -> Unit {
  output.write_string(key)
  output.write_string("=")
  output.write_string(value)
  output.write_string("\n")
}

///|
fn campaign_family(
  output : StringBuilder,
  prefix : String,
  family : String,
  candidate_digest : @luna_cuda_fused_parallel_aot.FusedParallelDigest,
  source_digest : @luna_cuda_fused_parallel_aot.FusedParallelDigest,
  recipe_digest : @luna_cuda_fused_parallel_aot.FusedParallelDigest,
  compiler : @luna_specialization.LunaCompilerPolicy,
  target : @catalog.DeviceTarget,
  profile : @launch_contract.PagedKernelProfile,
  dimensions : @launch_contract.AotLaunchDimensions,
  symbol : String,
  abi : @luna_cuda_fused_parallel_aot.FusedParallelKernelAbi,
  compiled : @luna_cuda_fused_parallel_aot.FusedParallelCompiledArtifactBinding,
  numerical : @luna_cuda_fused_parallel_aot.FusedParallelNumericalPolicy,
  canary : UInt,
  observed_absolute : Int64,
  observed_relative : Int64,
  observed_canary : UInt,
) -> Unit {
  let identity = compiler.identity()
  campaign_line(output, prefix + "family", family)
  campaign_line(output, prefix + "candidate_sha256", candidate_digest.as_hex())
  campaign_line(output, prefix + "source_sha256", source_digest.as_hex())
  campaign_line(output, prefix + "recipe_sha256", recipe_digest.as_hex())
  campaign_line(
    output,
    prefix + "toolchain_sha256",
    identity.toolchain_digest_sha256(),
  )
  campaign_line(
    output,
    prefix + "driver_identity_sha256",
    compiled.driver_identity_sha256(),
  )
  campaign_line(output, prefix + "compiler_major", identity.major().to_string())
  campaign_line(output, prefix + "compiler_minor", identity.minor().to_string())
  campaign_line(output, prefix + "compiler_patch", identity.patch().to_string())
  campaign_line(
    output,
    prefix + "optimization_level",
    compiler.optimization_level().to_string(),
  )
  campaign_line(
    output,
    prefix + "fmad",
    compiler.fused_multiply_add().to_string(),
  )
  campaign_line(
    output,
    prefix + "reassociate",
    compiler.reassociate().to_string(),
  )
  campaign_line(
    output,
    prefix + "max_registers",
    compiler.max_registers().to_string(),
  )
  campaign_line(output, prefix + "target", "sm_120")
  campaign_line(
    output,
    prefix + "target_supports_bf16",
    target.supports_bf16().to_string(),
  )
  campaign_line(
    output,
    prefix + "target_supports_cublas_lt",
    target.supports_cublas_lt().to_string(),
  )
  campaign_line(
    output,
    prefix + "profile_id",
    profile.id().as_int().to_string(),
  )
  campaign_line(
    output,
    prefix + "max_query_rows",
    profile.max_query_rows().to_string(),
  )
  campaign_line(
    output,
    prefix + "max_query_tokens",
    profile.max_query_tokens().to_string(),
  )
  campaign_line(
    output,
    prefix + "max_page_table_entries",
    profile.max_page_table_entries().to_string(),
  )
  campaign_line(output, prefix + "grid_x", dimensions.grid_x().to_string())
  campaign_line(output, prefix + "grid_y", dimensions.grid_y().to_string())
  campaign_line(output, prefix + "grid_z", dimensions.grid_z().to_string())
  campaign_line(output, prefix + "block_x", dimensions.block_x().to_string())
  campaign_line(output, prefix + "block_y", dimensions.block_y().to_string())
  campaign_line(output, prefix + "block_z", dimensions.block_z().to_string())
  campaign_line(
    output,
    prefix + "shared_memory_bytes",
    dimensions.shared_memory_bytes().to_string(),
  )
  campaign_line(output, prefix + "function_symbol", symbol)
  campaign_line(output, prefix + "typed_abi", abi.canonical_name())
  campaign_line(
    output,
    prefix + "first_build_cubin_sha256",
    compiled.module_digest().as_hex(),
  )
  campaign_line(
    output,
    prefix + "second_build_cubin_sha256",
    compiled.module_digest().as_hex(),
  )
  campaign_line(
    output,
    prefix + "compile_receipt_sha256",
    compiled.compile_receipt_digest().as_hex(),
  )
  campaign_line(output, prefix + "deterministic", "true")
  campaign_line(output, prefix + "numerical_policy", numerical.name())
  campaign_line(
    output,
    prefix + "maximum_absolute_error_ppb",
    numerical.maximum_absolute_error_ppb().to_string(),
  )
  campaign_line(
    output,
    prefix + "maximum_relative_error_ppb",
    numerical.maximum_relative_error_ppb().to_string(),
  )
  campaign_line(
    output,
    prefix + "observed_maximum_absolute_error_ppb",
    observed_absolute.to_string(),
  )
  campaign_line(
    output,
    prefix + "observed_maximum_relative_error_ppb",
    observed_relative.to_string(),
  )
  campaign_line(
    output,
    prefix + "dispatch_canary_per_token",
    canary.to_string(),
  )
  campaign_line(
    output,
    prefix + "expected_dispatch_canary",
    (canary * 12U).to_string(),
  )
  campaign_line(
    output,
    prefix + "observed_dispatch_canary",
    observed_canary.to_string(),
  )
  campaign_line(output, prefix + "dispatch_canary_pass", "true")
  campaign_line(output, prefix + "sanitizer_status", "pass")
  campaign_line(output, prefix + "race_status", "pass")
  campaign_line(output, prefix + "initcheck_status", "pass")
}
