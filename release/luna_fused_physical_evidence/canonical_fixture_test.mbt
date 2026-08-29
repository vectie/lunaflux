///|
fn fused_physical_test_line(
  output : StringBuilder,
  key : String,
  value : String,
) -> Unit {
  output.write_string(key)
  output.write_string("=")
  output.write_string(value)
  output.write_string("\n")
}

///|
fn fused_physical_test_family(
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
) -> Unit {
  let identity = compiler.identity()
  fused_physical_test_line(output, prefix + "family", family)
  fused_physical_test_line(
    output,
    prefix + "candidate_sha256",
    candidate_digest.as_hex(),
  )
  fused_physical_test_line(
    output,
    prefix + "source_sha256",
    source_digest.as_hex(),
  )
  fused_physical_test_line(
    output,
    prefix + "recipe_sha256",
    recipe_digest.as_hex(),
  )
  fused_physical_test_line(
    output,
    prefix + "toolchain_sha256",
    identity.toolchain_digest_sha256(),
  )
  fused_physical_test_line(
    output,
    prefix + "driver_identity_sha256",
    compiled.driver_identity_sha256(),
  )
  fused_physical_test_line(
    output,
    prefix + "compiler_major",
    identity.major().to_string(),
  )
  fused_physical_test_line(
    output,
    prefix + "compiler_minor",
    identity.minor().to_string(),
  )
  fused_physical_test_line(
    output,
    prefix + "compiler_patch",
    identity.patch().to_string(),
  )
  fused_physical_test_line(
    output,
    prefix + "optimization_level",
    compiler.optimization_level().to_string(),
  )
  fused_physical_test_line(
    output,
    prefix + "fmad",
    compiler.fused_multiply_add().to_string(),
  )
  fused_physical_test_line(
    output,
    prefix + "reassociate",
    compiler.reassociate().to_string(),
  )
  fused_physical_test_line(
    output,
    prefix + "max_registers",
    compiler.max_registers().to_string(),
  )
  fused_physical_test_line(output, prefix + "target", "sm_120")
  fused_physical_test_line(
    output,
    prefix + "target_supports_bf16",
    target.supports_bf16().to_string(),
  )
  fused_physical_test_line(
    output,
    prefix + "target_supports_cublas_lt",
    target.supports_cublas_lt().to_string(),
  )
  fused_physical_test_line(
    output,
    prefix + "profile_id",
    profile.id().as_int().to_string(),
  )
  fused_physical_test_line(
    output,
    prefix + "max_query_rows",
    profile.max_query_rows().to_string(),
  )
  fused_physical_test_line(
    output,
    prefix + "max_query_tokens",
    profile.max_query_tokens().to_string(),
  )
  fused_physical_test_line(
    output,
    prefix + "max_page_table_entries",
    profile.max_page_table_entries().to_string(),
  )
  fused_physical_test_line(
    output,
    prefix + "grid_x",
    dimensions.grid_x().to_string(),
  )
  fused_physical_test_line(
    output,
    prefix + "grid_y",
    dimensions.grid_y().to_string(),
  )
  fused_physical_test_line(
    output,
    prefix + "grid_z",
    dimensions.grid_z().to_string(),
  )
  fused_physical_test_line(
    output,
    prefix + "block_x",
    dimensions.block_x().to_string(),
  )
  fused_physical_test_line(
    output,
    prefix + "block_y",
    dimensions.block_y().to_string(),
  )
  fused_physical_test_line(
    output,
    prefix + "block_z",
    dimensions.block_z().to_string(),
  )
  fused_physical_test_line(
    output,
    prefix + "shared_memory_bytes",
    dimensions.shared_memory_bytes().to_string(),
  )
  fused_physical_test_line(output, prefix + "function_symbol", symbol)
  fused_physical_test_line(output, prefix + "typed_abi", abi.canonical_name())
  fused_physical_test_line(
    output,
    prefix + "first_build_cubin_sha256",
    compiled.module_digest().as_hex(),
  )
  fused_physical_test_line(
    output,
    prefix + "second_build_cubin_sha256",
    compiled.module_digest().as_hex(),
  )
  fused_physical_test_line(
    output,
    prefix + "compile_receipt_sha256",
    compiled.compile_receipt_digest().as_hex(),
  )
  fused_physical_test_line(output, prefix + "deterministic", "true")
  fused_physical_test_line(
    output,
    prefix + "numerical_policy",
    numerical.name(),
  )
  fused_physical_test_line(
    output,
    prefix + "maximum_absolute_error_ppb",
    numerical.maximum_absolute_error_ppb().to_string(),
  )
  fused_physical_test_line(
    output,
    prefix + "maximum_relative_error_ppb",
    numerical.maximum_relative_error_ppb().to_string(),
  )
  fused_physical_test_line(
    output,
    prefix + "observed_maximum_absolute_error_ppb",
    observed_absolute.to_string(),
  )
  fused_physical_test_line(
    output,
    prefix + "observed_maximum_relative_error_ppb",
    observed_relative.to_string(),
  )
  fused_physical_test_line(
    output,
    prefix + "dispatch_canary_per_token",
    canary.to_string(),
  )
  let aggregate_canary = canary * 12U
  fused_physical_test_line(
    output,
    prefix + "expected_dispatch_canary",
    aggregate_canary.to_string(),
  )
  fused_physical_test_line(
    output,
    prefix + "observed_dispatch_canary",
    aggregate_canary.to_string(),
  )
  fused_physical_test_line(output, prefix + "dispatch_canary_pass", "true")
  fused_physical_test_line(output, prefix + "sanitizer_status", "pass")
  fused_physical_test_line(output, prefix + "race_status", "pass")
  fused_physical_test_line(output, prefix + "initcheck_status", "pass")
}

///|
fn fused_physical_test_canonical(
  fixture : FusedPhysicalBlackBoxFixture,
) -> Bytes raise {
  let qkv = fixture.qkv
  let residual = fixture.residual
  let output = StringBuilder()
  fused_physical_test_line(
    output, "schema", "lunaflux-fused-physical-qualification-evidence.v1",
  )
  fused_physical_test_line(output, "family_count", "2")
  fused_physical_test_line(
    output,
    "referee_shape_set_sha256",
    fused_physical_test_referee_digest().as_hex(),
  )
  fused_physical_test_line(output, "referee_shape_count", "4")
  fused_physical_test_line(output, "qkv_absolute_tolerance_ppb", "250000")
  fused_physical_test_line(output, "qkv_relative_tolerance_ppb", "500000")
  fused_physical_test_line(output, "residual_absolute_tolerance_ppb", "100000")
  fused_physical_test_line(output, "residual_relative_tolerance_ppb", "250000")
  fused_physical_test_line(output, "artifact_directory_sealed", "true")
  fused_physical_test_line(
    output,
    "artifact_directory_seal_sha256",
    "e".repeat(64),
  )
  fused_physical_test_line(output, "evidence_directory_sealed", "true")
  fused_physical_test_line(
    output,
    "evidence_directory_seal_sha256",
    "f".repeat(64),
  )
  for
    field in [
      "stderr_bytes", "contexts_live_after", "streams_live_after", "events_live_after",
      "graphs_live_after", "graph_execs_live_after", "allocations_live_after", "modules_live_after",
      "device_bytes_live_after", "pending_work_after",
    ] {
    fused_physical_test_line(output, field, "0")
  }
  fused_physical_test_line(output, "manifest_bindable", "false")
  fused_physical_test_line(output, "promotion_authority", "absent")
  fused_physical_test_audit(output, qkv)
  fused_physical_test_family(
    output,
    "qkv_",
    "qkv-positioned-rope-paged-kvwrite",
    qkv.candidate_digest(),
    qkv.source_digest(),
    qkv.recipe_digest(),
    qkv.compiler_policy(),
    qkv.target(),
    qkv.profile(),
    qkv.launch_dimensions(),
    qkv.function_symbol(),
    QkvPositionedRopePagedKvWriteAbiV1,
    fixture.qkv_compiled,
    qkv.numerical_policy(),
    qkv.dispatch_canary(),
    1000L,
    2000L,
  )
  fused_physical_test_family(
    output,
    "residual_",
    "residual-rmsnorm",
    residual.candidate_digest(),
    residual.source_digest(),
    residual.recipe_digest(),
    residual.compiler_policy(),
    residual.target(),
    residual.profile(),
    residual.launch_dimensions(),
    residual.function_symbol(),
    ResidualRmsNormAbiV1,
    fixture.residual_compiled,
    residual.numerical_policy(),
    residual.dispatch_canary(),
    500L,
    1000L,
  )
  @utf8.encode(output.to_string())
}

///|
fn fused_physical_test_audit(
  output : StringBuilder,
  qkv : @luna_cuda_fused_parallel_aot.FusedQkvRopeKvWriteCudaCandidate,
) -> Unit {
  for
    pair in [
      ("schema", "lunaflux-fused-approved-device-observation.v1"),
      ("device_uuid", "GPU-test-sm120"),
      ("device_pci", "00000000:01:00.0"),
      ("device_name_sha256", "6".repeat(64)),
      ("device_total_memory_bytes", "17179869184"),
      ("device_compute", "12.0"),
      ("host_driver_version", "590.48.01"),
      ("device_identity_record_sha256", "7".repeat(64)),
      ("approved_policy_sha256", "9".repeat(64)),
      ("nvcc_sha256", "8".repeat(64)),
      ("compute_sanitizer_sha256", "a".repeat(64)),
    ] {
    fused_physical_test_line(output, pair.0, pair.1)
  }
  for
    pair in [
      ("referee_source_sha256", "1".repeat(64)),
      ("probe_executable_sha256", "2".repeat(64)),
      (
        "standalone_qkv_source_sha256",
        qkv.qkv_fallback().source_digest().as_hex(),
      ),
      (
        "standalone_qkv_recipe_sha256",
        qkv.qkv_fallback().recipe_digest().as_hex(),
      ),
      ("standalone_qkv_cubin_sha256", "3".repeat(64)),
      ("standalone_qkv_receipt_sha256", "3".repeat(64)),
      (
        "standalone_rope_source_sha256",
        qkv.rope_fallback().source_digest().as_hex(),
      ),
      (
        "standalone_rope_recipe_sha256",
        qkv.rope_fallback().recipe_digest().as_hex(),
      ),
      ("standalone_rope_cubin_sha256", "3".repeat(64)),
      ("standalone_rope_receipt_sha256", "3".repeat(64)),
      (
        "standalone_kv_write_source_sha256",
        qkv.kv_write_fallback().source_digest().as_hex(),
      ),
      (
        "standalone_kv_write_recipe_sha256",
        qkv.kv_write_fallback().recipe_digest().as_hex(),
      ),
      ("standalone_kv_write_cubin_sha256", "3".repeat(64)),
      ("standalone_kv_write_receipt_sha256", "3".repeat(64)),
      ("standalone_oracle_output_sha256", "3".repeat(64)),
    ] {
    fused_physical_test_line(output, pair.0, pair.1)
  }
  fused_physical_test_line(output, "standalone_oracle_executed", "true")
  for
    key in [
      "ptxas_sha256", "cuobjdump_sha256", "nvdisasm_sha256", "ptxas_version_sha256",
      "cuobjdump_version_sha256", "nvdisasm_version_sha256",
    ] {
    fused_physical_test_line(output, key, "3".repeat(64))
  }
  fused_physical_test_resource(output, "qkv_", 0)
  fused_physical_test_resource(output, "residual_", 512)
}

///|
fn fused_physical_test_resource(
  output : StringBuilder,
  prefix : String,
  static_shared : Int,
) -> Unit {
  fused_physical_test_line(
    output,
    prefix + "cuobjdump_sass_sha256",
    "4".repeat(64),
  )
  fused_physical_test_line(
    output,
    prefix + "cuobjdump_sass_instruction_count",
    "17",
  )
  fused_physical_test_line(
    output,
    prefix + "nvdisasm_sass_sha256",
    "5".repeat(64),
  )
  fused_physical_test_line(
    output,
    prefix + "nvdisasm_sass_instruction_count",
    "17",
  )
  fused_physical_test_line(
    output,
    prefix + "global_load_instruction_count",
    "4",
  )
  fused_physical_test_line(
    output,
    prefix + "global_store_instruction_count",
    "3",
  )
  fused_physical_test_line(output, prefix + "registers_per_thread", "64")
  fused_physical_test_line(
    output,
    prefix + "static_shared_bytes",
    static_shared.to_string(),
  )
  for
    suffix in [
      "local_bytes", "stack_bytes", "spill_store_bytes", "spill_load_bytes",
    ] {
    fused_physical_test_line(output, prefix + suffix, "0")
  }
}

///|
fn fused_physical_test_admit(
  fixture : FusedPhysicalBlackBoxFixture,
  canonical : Bytes,
) -> AdmittedFusedPhysicalEvidence raise {
  admit_fused_physical_evidence(
    fixture.qkv,
    fixture.qkv_compiled,
    fixture.residual,
    fixture.residual_compiled,
    canonical,
    expected_digest=fused_physical_test_digest(canonical),
    expected_approved_policy_digest=fused_physical_test_policy_digest(),
  )
}
