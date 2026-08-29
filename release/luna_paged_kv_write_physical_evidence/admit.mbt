///|
fn kv_require_digest(
  reader : PagedKvWritePhysicalReader,
  key : String,
  expected : @luna_cuda_paged_kv_write_aot.PagedKvWriteDigest,
  rule : PagedKvWritePhysicalRule,
) -> Unit raise PagedKvWritePhysicalError {
  if reader.digest(key) != expected {
    raise Mismatch(rule)
  }
}

///|
fn kv_require_text(
  reader : PagedKvWritePhysicalReader,
  key : String,
  expected : String,
  rule : PagedKvWritePhysicalRule,
) -> Unit raise PagedKvWritePhysicalError {
  reader.literal(key, expected, rule)
}

///|
fn kv_require_int(
  reader : PagedKvWritePhysicalReader,
  key : String,
  expected : Int64,
  rule : PagedKvWritePhysicalRule,
) -> Unit raise PagedKvWritePhysicalError {
  if reader.int64(key) != expected {
    raise Mismatch(rule)
  }
}

///|
fn kv_admit_tools_and_device(
  reader : PagedKvWritePhysicalReader,
  expected_policy : @luna_cuda_paged_kv_write_aot.PagedKvWriteDigest,
) -> (String, String, String) raise PagedKvWritePhysicalError {
  kv_require_digest(reader, "approved_policy_sha256", expected_policy, Policy)
  let device_uuid = reader.value("device_uuid")
  let device_pci = reader.value("device_pci")
  ignore(reader.value("host_driver_version"))
  ignore(reader.digest("device_name_sha256"))
  if reader.int64("device_total_memory_bytes") <= 0L {
    raise Mismatch(Target)
  }
  reader.literal("device_compute", "12.0", Target)
  if reader.int64("cuda_driver_version") <= 0L {
    raise Mismatch(Target)
  }
  kv_require_text(reader, "context_device_uuid", device_uuid, Target)
  kv_require_text(reader, "context_device_pci", device_pci, Target)
  for
    key in [
      "nvcc_sha256", "ptxas_sha256", "cuobjdump_sha256", "nvdisasm_sha256", "compute_sanitizer_sha256",
      "nvidia_smi_sha256", "driver_record_sha256", "nvcc_version_sha256", "ptxas_version_sha256",
      "cuobjdump_version_sha256", "nvdisasm_version_sha256", "compute_sanitizer_version_sha256",
    ] {
    ignore(reader.digest(key))
  }
  let driver = reader.value("driver_identity_sha256")
  ignore(
    @luna_cuda_paged_kv_write_aot.PagedKvWriteDigest::from_sha256(driver) catch {
      _ => raise Invalid(Digest)
    },
  )
  (driver, device_uuid, device_pci)
}

///|
fn kv_admit_candidate(
  reader : PagedKvWritePhysicalReader,
  candidate : @luna_cuda_paged_kv_write_aot.PagedKvWriteCudaCandidate,
  compiled : @luna_cuda_paged_kv_write_aot.PagedKvWriteCompiledArtifact,
  driver : String,
) -> Unit raise PagedKvWritePhysicalError {
  kv_require_digest(
    reader,
    "candidate_sha256",
    candidate.candidate_digest(),
    Candidate,
  )
  kv_require_digest(
    reader,
    "source_sha256",
    candidate.source_digest(),
    Candidate,
  )
  kv_require_digest(
    reader,
    "recipe_sha256",
    candidate.recipe_digest(),
    Candidate,
  )
  kv_require_digest(
    reader,
    "compiled_binding_sha256",
    compiled.binding_digest(),
    Compiled,
  )
  kv_require_digest(reader, "module_sha256", compiled.module_digest(), Compiled)
  kv_require_digest(
    reader,
    "compile_receipt_sha256",
    compiled.compile_receipt_digest(),
    Compiled,
  )
  ignore(reader.digest("serial_oracle_source_sha256"))
  ignore(reader.digest("scalar_referee_source_sha256"))
  ignore(reader.digest("probe_executable_sha256"))
  kv_require_text(
    reader,
    "model_content_sha256",
    candidate.model_content_digest_sha256(),
    Candidate,
  )
  kv_require_text(
    reader,
    "model_plan_sha256",
    candidate.model_plan_digest_sha256(),
    Candidate,
  )
  kv_require_int(
    reader,
    "attention_operation",
    candidate.operation_id().as_int().to_int64(),
    Candidate,
  )
  kv_require_int(
    reader,
    "positioned_qkv_activation",
    candidate.positioned_qkv_activation().as_int().to_int64(),
    Candidate,
  )
  kv_require_int(
    reader,
    "query_heads",
    candidate.query_heads().to_int64(),
    Candidate,
  )
  kv_require_int(reader, "kv_heads", candidate.kv_heads().to_int64(), Candidate)
  kv_require_int(
    reader,
    "head_dimension",
    candidate.head_dimension().to_int64(),
    Candidate,
  )
  let row_width = (candidate.query_heads() + 2 * candidate.kv_heads()) *
    candidate.head_dimension()
  kv_require_int(
    reader,
    "positioned_qkv_row_width",
    row_width.to_int64(),
    Candidate,
  )
  reader.literal("target", "sm_120", Target)
  let target = candidate.target()
  if target.compute_major() != 12 ||
    target.compute_minor() != 0 ||
    compiled.target() != target {
    raise Mismatch(Target)
  }
  let profile = candidate.profile()
  kv_require_int(
    reader,
    "max_query_rows",
    profile.max_query_rows().to_int64(),
    Candidate,
  )
  kv_require_int(
    reader,
    "max_query_tokens",
    profile.max_query_tokens().to_int64(),
    Candidate,
  )
  kv_require_int(
    reader,
    "max_page_table_entries",
    profile.max_page_table_entries().to_int64(),
    Candidate,
  )
  kv_require_int(
    reader,
    "tokens_per_page",
    candidate.tokens_per_page().to_int64(),
    Candidate,
  )
  kv_require_int(
    reader,
    "total_page_count",
    candidate.total_page_count().to_int64(),
    Candidate,
  )
  kv_require_int(
    reader,
    "page_stride_bytes",
    candidate.page_stride_bytes(),
    Candidate,
  )
  let compiler = candidate.compiler_policy()
  kv_require_text(
    reader,
    "toolchain_sha256",
    compiler.identity().toolchain_digest_sha256(),
    Candidate,
  )
  reader.literal("compiler_version", "13.1.115", Candidate)
  if compiler.identity().major() != 13 ||
    compiler.identity().minor() != 1 ||
    compiler.identity().patch() != 115 {
    raise Mismatch(Candidate)
  }
  kv_require_text(
    reader,
    "function_symbol",
    candidate.function_symbol(),
    Candidate,
  )
  kv_require_text(
    reader,
    "raw_pointer_abi",
    candidate.kernel_abi().canonical_name(),
    Candidate,
  )
  if compiled.function_symbol() != candidate.function_symbol() ||
    compiled.kernel_abi() != candidate.kernel_abi() ||
    compiled.driver_identity_sha256() != driver {
    raise Mismatch(Compiled)
  }
  let dimensions = candidate.launch_dimensions()
  for
    pair in [
      ("grid_x", dimensions.grid_x()),
      ("grid_y", dimensions.grid_y()),
      ("grid_z", dimensions.grid_z()),
      ("block_x", dimensions.block_x()),
      ("block_y", dimensions.block_y()),
      ("block_z", dimensions.block_z()),
      ("shared_memory_bytes", dimensions.shared_memory_bytes()),
    ] {
    kv_require_int(reader, pair.0, pair.1.to_int64(), Candidate)
  }
  kv_require_int(
    reader,
    "dispatch_canary_per_token",
    candidate.dispatch_canary_per_token().to_int64(),
    Canary,
  )
  kv_require_text(
    reader,
    "fallback_source_sha256",
    candidate.fallback_source_digest_sha256(),
    Candidate,
  )
  kv_require_text(
    reader,
    "fallback_recipe_sha256",
    candidate.fallback_recipe_digest_sha256(),
    Candidate,
  )
}

///|
fn kv_admit_observation(
  reader : PagedKvWritePhysicalReader,
  candidate : @luna_cuda_paged_kv_write_aot.PagedKvWriteCudaCandidate,
) -> Unit raise PagedKvWritePhysicalError {
  kv_require_int(reader, "case_count", 6L, Oracle)
  for
    key in [
      "prefill_case_pass", "decode_case_pass", "mixed_case_pass", "origin_case_pass",
      "page_tail_case_pass", "cross_page_case_pass", "multirow_case_pass", "full_grid_launch_pass",
      "expected_cells_pass", "byte_oracle_pass", "scalar_oracle_pass", "serial_cuda_oracle_pass",
      "guards_unchanged", "non_target_cache_unchanged", "dispatch_canary_exact",
      "deterministic_cubins", "memcheck_status", "racecheck_status", "initcheck_status",
    ] {
    reader.literal(key, "pass", Oracle)
  }
  let tokens = reader.int64("qualified_token_count")
  let kv_width = reader.int64("kv_width")
  if tokens <= 0L ||
    kv_width != candidate.kv_width().to_int64() ||
    tokens > candidate.profile().max_query_tokens().to_int64() * 6L {
    raise Mismatch(Oracle)
  }
  let expected_writes = tokens * kv_width
  kv_require_int(reader, "expected_key_mutations", expected_writes, Oracle)
  kv_require_int(reader, "observed_key_mutations", expected_writes, Oracle)
  kv_require_int(reader, "expected_value_mutations", expected_writes, Oracle)
  kv_require_int(reader, "observed_value_mutations", expected_writes, Oracle)
  kv_require_int(reader, "dispatch_canary_cell_count", tokens, Canary)
  let expected_canary = tokens *
    candidate.dispatch_canary_per_token().to_int64()
  if expected_canary > 4294967295L {
    raise Mismatch(Canary)
  }
  kv_require_int(reader, "expected_dispatch_canary", expected_canary, Canary)
  kv_require_int(reader, "observed_dispatch_canary", expected_canary, Canary)
  ignore(reader.digest("cuobjdump_sass_sha256"))
  let cu_count = reader.int64("cuobjdump_sass_instruction_count")
  ignore(reader.digest("nvdisasm_sass_sha256"))
  let nv_count = reader.int64("nvdisasm_sass_instruction_count")
  reader.literal("sass_global_load_observed", "true", Resource)
  reader.literal("sass_global_store_observed", "true", Resource)
  let registers = reader.int64("registers_per_thread")
  if cu_count <= 0L ||
    cu_count != nv_count ||
    registers <= 0L ||
    registers > 128L {
    raise Mismatch(Resource)
  }
  for
    key in [
      "static_shared_bytes", "local_bytes", "stack_bytes", "spill_store_bytes", "spill_load_bytes",
      "contexts_live_after", "streams_live_after", "allocations_live_after", "modules_live_after",
      "functions_live_after", "device_bytes_live_after", "pending_work_after", "stderr_bytes",
    ] {
    kv_require_int(reader, key, 0L, Resource)
  }
}

///|
pub fn admit_qualified_paged_kv_write_artifact(
  candidate : @luna_cuda_paged_kv_write_aot.PagedKvWriteCudaCandidate,
  compiled : @luna_cuda_paged_kv_write_aot.PagedKvWriteCompiledArtifact,
  canonical : Bytes,
  expected_evidence_digest~ : @luna_cuda_paged_kv_write_aot.PagedKvWriteDigest,
  expected_approved_policy_digest~ : @luna_cuda_paged_kv_write_aot.PagedKvWriteDigest,
) -> QualifiedPagedKvWriteArtifact raise PagedKvWritePhysicalError {
  if candidate.manifest_bindable() ||
    candidate.promotion_authority() ||
    compiled.manifest_bindable() ||
    compiled.promotion_authority() {
    raise Mismatch(Authority)
  }
  let evidence_digest = kv_physical_sha256(canonical)
  if evidence_digest != expected_evidence_digest {
    raise Mismatch(Digest)
  }
  let reader = kv_physical_reader(canonical)
  reader.literal(
    "schema",
    "lunaflux-paged-kv-write-physical-evidence.v1",
    Canonical,
  )
  let device = kv_admit_tools_and_device(
    reader, expected_approved_policy_digest,
  )
  kv_admit_candidate(reader, candidate, compiled, device.0)
  kv_admit_observation(reader, candidate)
  reader.literal("physical_cuda_observed", "true", Authority)
  reader.literal("synthetic_test_only", "false", Authority)
  let files_manifest_digest = reader.digest("files_manifest_sha256")
  let outer_seal_digest = reader.digest("outer_seal_sha256")
  reader.literal("manifest_bindable", "false", Authority)
  reader.literal("promotion_authority", "absent", Authority)
  reader.literal("runtime_dispatch_authority", "absent", Authority)
  reader.finish()
  {
    candidate,
    compiled,
    evidence_digest,
    approved_policy_digest: expected_approved_policy_digest,
    device_uuid: device.1,
    device_pci: device.2,
    files_manifest_digest,
    outer_seal_digest,
  }
}
