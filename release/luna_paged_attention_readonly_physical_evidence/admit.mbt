///|
pub fn admit_paged_attention_readonly_physical_evidence(
  candidate : @luna_cuda_paged_attention_readonly_aot.PagedAttentionReadOnlyCudaCandidate,
  compiled : @luna_cuda_paged_attention_readonly_aot.PagedAttentionReadOnlyCompiledBinding,
  approved_policy : ApprovedPagedAttentionReadOnlyPhysicalPolicy,
  result : Bytes,
  files_manifest : Bytes,
  outer_seal : Bytes,
  expected_outer_seal_digest : PagedAttentionReadOnlyPhysicalEvidenceDigest,
  limits : PagedAttentionReadOnlyPhysicalEvidenceLimits,
) -> AdmittedPagedAttentionReadOnlyPhysicalEvidence raise PagedAttentionReadOnlyPhysicalEvidenceError {
  if candidate.manifest_bindable() ||
    candidate.promotion_authorized() ||
    compiled.manifest_bindable() ||
    compiled.promotion_authorized() ||
    compiled.candidate_digest() != candidate.candidate_digest() ||
    compiled.source_digest() != candidate.source_digest() ||
    compiled.recipe_digest() != candidate.recipe_digest() {
    raise PagedAttentionReadOnlyPhysicalEvidenceMismatch(Candidate)
  }
  let (manifest_digest, result_digest) = readonly_physical_outer_seal(
    result, files_manifest, outer_seal, expected_outer_seal_digest,
  )
  let reader = readonly_physical_reader(result, limits.maximum_result_bytes)
  reader.literal(
    "schema",
    "lunaflux-paged-attention-readonly-physical-campaign.v1",
    Canonical,
  )
  reader.literal(
    "outcome",
    "paged-attention-readonly-physical-campaign-pass",
    Target,
  )
  reader.digest_text(
    "approved_policy_sha256",
    approved_policy.digest.as_hex(),
    Toolchain,
  )
  reader.digest_text(
    "candidate_sha256",
    candidate.candidate_digest().as_hex(),
    Candidate,
  )
  reader.digest_text(
    "source_sha256",
    candidate.source_digest().as_hex(),
    Candidate,
  )
  reader.digest_text(
    "recipe_sha256",
    candidate.recipe_digest().as_hex(),
    Candidate,
  )
  let fallback = candidate.reference_fallback()
  reader.digest_text(
    "fallback_source_sha256",
    fallback.source_digest().as_hex(),
    Fallback,
  )
  reader.digest_text(
    "fallback_recipe_sha256",
    fallback.recipe_digest().as_hex(),
    Fallback,
  )
  reader.digest_text(
    "compiled_binding_sha256",
    compiled.binding_digest().as_hex(),
    CompiledBinding,
  )
  let cubin = reader.digest("cubin_sha256")
  if cubin.as_hex() != compiled.module_digest().as_hex() {
    raise PagedAttentionReadOnlyPhysicalEvidenceMismatch(CompiledBinding)
  }
  let receipt = reader.digest("compile_receipt_sha256")
  if receipt.as_hex() != compiled.compile_receipt_digest().as_hex() {
    raise PagedAttentionReadOnlyPhysicalEvidenceMismatch(CompiledBinding)
  }
  reader.exact_int("operation_id", candidate.operation_id().as_int(), Candidate)
  reader.literal("target", "sm_120", Target)
  let profile = candidate.profile()
  reader.literal(
    "profile",
    profile.max_query_rows().to_string() +
    "," +
    profile.max_query_tokens().to_string() +
    "," +
    profile.max_page_table_entries().to_string(),
    Candidate,
  )
  reader.literal("kv_layout", "layer_major_split_key_value_v1", Layout)
  reader.exact_int("kv_layout_stable_version", 1, Layout)
  reader.exact_int("tokens_per_page", fallback.tokens_per_page(), Layout)
  reader.exact_int(
    "page_stride_bytes",
    fallback.page_stride_bytes().to_int(),
    Layout,
  )
  reader.exact_int(
    "physical_page_capacity",
    fallback.physical_page_capacity().to_int(),
    Layout,
  )
  reader.exact_int("input_row_width", candidate.input_row_width(), Candidate)
  reader.exact_int(
    "input_activation_ref",
    candidate.input_activation().as_int(),
    Candidate,
  )
  reader.literal("function_symbol", candidate.function_symbol(), Candidate)
  reader.literal(
    "raw_pointer_abi",
    candidate.kernel_abi().canonical_name(),
    Candidate,
  )
  reader.literal(
    "selection_precondition",
    "standalone-positioned-rope-paged-kvwrite-complete-v1",
    Candidate,
  )
  let canary = reader.int("dispatch_canary_per_token")
  let canary_cells = reader.int("dispatch_canary_cell_count")
  if canary != candidate.dispatch_canary_per_token() ||
    canary_cells != profile.max_query_tokens() ||
    canary <= 0 ||
    canary.to_int64() * canary_cells.to_int64() > 4294967295L {
    raise PagedAttentionReadOnlyPhysicalEvidenceMismatch(Canary)
  }
  reader.literal(
    "dispatch_canary_publication",
    "exactly-once-after-output",
    Canary,
  )
  reader.literal("dispatch_canary_exact", "true", Canary)
  reader.literal("dispatch_canary_tail_zero", "true", Canary)
  reader.literal("dispatch_canary_sum_checked", "true", Canary)
  reader.literal("kv_cache_mutation", "none", CacheImmutability)
  let driver_identity = reader.digest("driver_identity_sha256")
  if driver_identity != approved_policy.driver_identity ||
    driver_identity.as_hex() !=
    fallback.compiler_policy().identity().toolchain_digest_sha256() {
    raise PagedAttentionReadOnlyPhysicalEvidenceMismatch(Toolchain)
  }
  let driver_record = reader.digest("driver_record_sha256")
  if driver_record != approved_policy.driver_record {
    raise PagedAttentionReadOnlyPhysicalEvidenceMismatch(Toolchain)
  }
  let device_record = reader.digest("device_record_sha256")
  reader.literal("compiler_version", "13.1.115", Toolchain)
  reader.literal("device_uuid", approved_policy.device_uuid, Target)
  reader.literal("device_pci_bus_id", approved_policy.device_pci_bus_id, Target)
  reader.literal("device_name_hex", approved_policy.device_name_hex, Target)
  reader.literal(
    "device_total_memory_bytes",
    approved_policy.device_total_memory_bytes.to_string(),
    Target,
  )
  if reader.digest("nvcc_sha256") != approved_policy.nvcc ||
    reader.digest("ptxas_sha256") != approved_policy.ptxas ||
    reader.digest("cuobjdump_sha256") != approved_policy.cuobjdump ||
    reader.digest("nvdisasm_sha256") != approved_policy.nvdisasm ||
    reader.digest("compute_sanitizer_sha256") != approved_policy.sanitizer {
    raise PagedAttentionReadOnlyPhysicalEvidenceMismatch(Toolchain)
  }
  let nvcc_version = reader.digest("nvcc_version_sha256")
  let ptxas_version = reader.digest("ptxas_version_sha256")
  let cuobjdump_version = reader.digest("cuobjdump_version_sha256")
  let nvdisasm_version = reader.digest("nvdisasm_version_sha256")
  let sanitizer_version = reader.digest("compute_sanitizer_version_sha256")
  let probe_source = reader.digest("probe_source_sha256")
  let probe_binary = reader.digest("probe_binary_sha256")
  let cu_sass = reader.digest("cuobjdump_sass_sha256")
  let cu_count = reader.int("cuobjdump_sass_instruction_count")
  let nv_sass = reader.digest("nvdisasm_sass_sha256")
  let nv_count = reader.int("nvdisasm_sass_instruction_count")
  if cu_count <= 0 || nv_count <= 0 {
    raise PagedAttentionReadOnlyPhysicalEvidenceMismatch(Sass)
  }
  reader.literal("sass_global_load_observed", "true", Sass)
  reader.literal("sass_global_store_observed", "true", Sass)
  let registers = reader.int("registers_per_thread")
  reader.exact_int("register_bound", 128, Resource)
  let shared = reader.int("static_shared_bytes")
  reader.exact_int("static_shared_bound", 0, Resource)
  let stack = reader.int("stack_bytes")
  let local_bytes = reader.int("local_bytes")
  let spill_store = reader.int("spill_store_bytes")
  let spill_load = reader.int("spill_load_bytes")
  if registers <= 0 ||
    registers > 128 ||
    shared != 0 ||
    stack != 0 ||
    local_bytes != 0 ||
    spill_store != 0 ||
    spill_load != 0 {
    raise PagedAttentionReadOnlyPhysicalEvidenceMismatch(Resource)
  }
  reader.literal("resource_bounds_passed", "true", Resource)
  reader.literal(
    "case_families",
    "origin,page-tail,cross-page,multirow,long-context",
    Numerical,
  )
  reader.literal(
    "scheduler_modes",
    "prefill-only,decode-only,mixed-prefill-decode",
    Numerical,
  )
  reader.exact_int("dispatch_grid_x", profile.max_query_tokens(), Candidate)
  let cases = reader.int("numeric_case_count")
  if cases < 36 || cases > 1152 || cases % 36 != 0 {
    raise PagedAttentionReadOnlyPhysicalEvidenceMismatch(Numerical)
  }
  reader.literal("output_dtype", "bf16", Numerical)
  reader.literal("absolute_tolerance", "0.0078125", Numerical)
  reader.literal("relative_tolerance", "0.01", Numerical)
  if reader.nonnegative_double("cpu_vs_candidate_max_abs_error") > 0.0078125 ||
    reader.nonnegative_double("serial_vs_candidate_max_abs_error") > 0.0078125 {
    raise PagedAttentionReadOnlyPhysicalEvidenceMismatch(Numerical)
  }
  reader.literal("cpu_oracle", "independent-ordered-f32-v1", Numerical)
  reader.literal(
    "serial_cuda_oracle",
    "independent-ordered-f32-kernel-v1",
    Numerical,
  )
  reader.exact_int("cache_snapshot_bytes", 81920, CacheImmutability)
  reader.literal("cache_snapshot_unchanged", "true", CacheImmutability)
  reader.literal("input_guards_unchanged", "true", CacheImmutability)
  reader.literal("output_guards_unchanged", "true", CacheImmutability)
  reader.literal("dispatch_symbol_resolved", "true", Candidate)
  let memcheck = reader.digest("memcheck_log_sha256")
  reader.exact_int("memcheck_error_summary_count", 1, Sanitizer)
  reader.exact_int("memcheck_errors", 0, Sanitizer)
  let racecheck = reader.digest("racecheck_log_sha256")
  reader.exact_int("racecheck_error_summary_count", 1, Sanitizer)
  reader.exact_int("racecheck_errors", 0, Sanitizer)
  let initcheck = reader.digest("initcheck_log_sha256")
  reader.exact_int("initcheck_error_summary_count", 1, Sanitizer)
  reader.exact_int("initcheck_errors", 0, Sanitizer)
  reader.literal(
    "cleanup_balance",
    "module0,allocation0,device_bytes0,pending0,cleanup0",
    Resource,
  )
  reader.literal("physical_cuda_observed", "true", Target)
  reader.literal("synthetic_test_only", "false", Target)
  reader.literal("qualification_only", "true", Authority)
  reader.literal("manifest_bindable", "false", Authority)
  reader.literal("promotion_authority", "absent", Authority)
  let recorded_manifest = reader.digest("evidence_files_manifest_sha256")
  if recorded_manifest != manifest_digest {
    raise PagedAttentionReadOnlyPhysicalEvidenceMismatch(Seal)
  }
  reader.finish()

  let entries = readonly_physical_manifest_entries(
    files_manifest,
    limits.maximum_manifest_bytes,
  )
  readonly_physical_manifest_require(
    entries,
    "policy/approved.v1",
    approved_policy.digest,
    Toolchain,
  )
  let candidate_digest = readonly_physical_read_digest(
    candidate.candidate_digest().as_hex(),
  )
  let source_digest = readonly_physical_read_digest(
    candidate.source_digest().as_hex(),
  )
  let recipe_digest = readonly_physical_read_digest(
    candidate.recipe_digest().as_hex(),
  )
  let fallback_source = readonly_physical_read_digest(
    fallback.source_digest().as_hex(),
  )
  let fallback_recipe = readonly_physical_read_digest(
    fallback.recipe_digest().as_hex(),
  )
  for
    requirement in [
      ("export/readonly.canonical", candidate_digest, Candidate),
      ("export/readonly.cu", source_digest, Candidate),
      ("export/readonly.recipe", recipe_digest, Candidate),
      ("export/fallback.cu", fallback_source, Fallback),
      ("export/fallback.recipe", fallback_recipe, Fallback),
      ("builds/first/kernel.cubin", cubin, CompiledBinding),
      ("builds/second/kernel.cubin", cubin, CompiledBinding),
      ("builds/resource/kernel.cubin", cubin, CompiledBinding),
      ("builds/first/kernel.cu", source_digest, Candidate),
      ("builds/second/kernel.cu", source_digest, Candidate),
      ("builds/resource/kernel.cu", source_digest, Candidate),
      ("measurements/compile-receipt.v1", receipt, CompiledBinding),
      ("measurements/device-identity.v1", device_record, Target),
      ("measurements/cuobjdump-sass.stdout", cu_sass, Sass),
      ("measurements/nvdisasm-sass.stdout", nv_sass, Sass),
      ("measurements/memcheck.log", memcheck, Sanitizer),
      ("measurements/racecheck.log", racecheck, Sanitizer),
      ("measurements/initcheck.log", initcheck, Sanitizer),
      ("probe/probe.cu", probe_source, Candidate),
      ("probe/physical_probe", probe_binary, Candidate),
    ] {
    readonly_physical_manifest_require(
      entries,
      requirement.0,
      requirement.1,
      requirement.2,
    )
  }
  readonly_physical_require_pair(
    entries,
    "measurements/driver-before.v1",
    "measurements/driver-after.v1",
    driver_record,
    Toolchain,
  )
  readonly_physical_require_pair(
    entries,
    "measurements/nvcc-version-before.stdout",
    "measurements/nvcc-version-after.stdout",
    nvcc_version,
    Toolchain,
  )
  readonly_physical_require_pair(
    entries,
    "measurements/ptxas-version-before.stdout",
    "measurements/ptxas-version-after.stdout",
    ptxas_version,
    Toolchain,
  )
  readonly_physical_require_pair(
    entries,
    "measurements/cuobjdump-version-before.stdout",
    "measurements/cuobjdump-version-after.stdout",
    cuobjdump_version,
    Toolchain,
  )
  readonly_physical_require_pair(
    entries,
    "measurements/nvdisasm-version-before.stdout",
    "measurements/nvdisasm-version-after.stdout",
    nvdisasm_version,
    Toolchain,
  )
  readonly_physical_require_pair(
    entries,
    "measurements/compute-sanitizer-version-before.stdout",
    "measurements/compute-sanitizer-version-after.stdout",
    sanitizer_version,
    Toolchain,
  )
  {
    candidate_digest: candidate.candidate_digest().as_hex(),
    compiled_binding_digest: compiled.binding_digest().as_hex(),
    cubin_digest: cubin,
    fallback_source_digest: fallback.source_digest().as_hex(),
    fallback_recipe_digest: fallback.recipe_digest().as_hex(),
    operation_id: candidate.operation_id(),
    input_activation: candidate.input_activation(),
    input_row_width: candidate.input_row_width(),
    dispatch_canary_per_token: canary,
    dispatch_canary_cell_count: canary_cells,
    numeric_case_count: cases,
    canonical: result,
    digest: result_digest,
    files_manifest_digest: manifest_digest,
    outer_seal_digest: expected_outer_seal_digest,
  }
}

///|
pub fn bind_qualified_paged_attention_readonly_artifact(
  candidate : @luna_cuda_paged_attention_readonly_aot.PagedAttentionReadOnlyCudaCandidate,
  compiled : @luna_cuda_paged_attention_readonly_aot.PagedAttentionReadOnlyCompiledBinding,
  evidence : AdmittedPagedAttentionReadOnlyPhysicalEvidence,
) -> QualifiedPagedAttentionReadOnlyArtifact raise PagedAttentionReadOnlyPhysicalEvidenceError {
  if candidate.candidate_digest().as_hex() != evidence.candidate_digest ||
    compiled.binding_digest().as_hex() != evidence.compiled_binding_digest ||
    compiled.module_digest().as_hex() != evidence.cubin_digest.as_hex() ||
    candidate.reference_fallback().source_digest().as_hex() !=
    evidence.fallback_source_digest ||
    candidate.reference_fallback().recipe_digest().as_hex() !=
    evidence.fallback_recipe_digest ||
    candidate.operation_id() != evidence.operation_id ||
    candidate.input_activation() != evidence.input_activation ||
    candidate.input_row_width() != evidence.input_row_width {
    raise PagedAttentionReadOnlyPhysicalEvidenceMismatch(Candidate)
  }
  { candidate, compiled, evidence }
}
