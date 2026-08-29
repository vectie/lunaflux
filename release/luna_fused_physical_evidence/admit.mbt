///|
fn fused_physical_admit_global(
  reader : FusedPhysicalReader,
  qkv : ExpectedFusedPhysicalFamily,
  residual : ExpectedFusedPhysicalFamily,
  expected_approved_policy_digest : @luna_cuda_fused_parallel_aot.FusedParallelDigest,
) -> (
  @luna_cuda_fused_parallel_aot.FusedParallelDigest,
  @luna_cuda_fused_parallel_aot.FusedParallelDigest,
  @luna_cuda_fused_parallel_aot.FusedParallelDigest,
) raise FusedPhysicalEvidenceError {
  reader.require_literal(
    "schema",
    "lunaflux-fused-physical-qualification-evidence.v1",
    CanonicalDocument,
  )
  reader.require_int("family_count", 2, Family)
  let referee_digest = reader.read_digest("referee_shape_set_sha256")
  if referee_digest != fused_physical_referee_shape_set_digest() {
    raise FusedPhysicalEvidenceMismatch(Referee)
  }
  reader.require_int(
    "referee_shape_count",
    FUSED_PHYSICAL_REFEREE_SHAPE_COUNT,
    Referee,
  )
  reader.require_int64(
    "qkv_absolute_tolerance_ppb",
    qkv.numerical_policy.maximum_absolute_error_ppb(),
    Referee,
  )
  reader.require_int64(
    "qkv_relative_tolerance_ppb",
    qkv.numerical_policy.maximum_relative_error_ppb(),
    Referee,
  )
  reader.require_int64(
    "residual_absolute_tolerance_ppb",
    residual.numerical_policy.maximum_absolute_error_ppb(),
    Referee,
  )
  reader.require_int64(
    "residual_relative_tolerance_ppb",
    residual.numerical_policy.maximum_relative_error_ppb(),
    Referee,
  )
  reader.require_literal("artifact_directory_sealed", "true", Seal)
  let artifact_seal = reader.read_digest("artifact_directory_seal_sha256")
  reader.require_literal("evidence_directory_sealed", "true", Seal)
  let evidence_seal = reader.read_digest("evidence_directory_seal_sha256")
  reader.require_int64("stderr_bytes", 0L, ResourceBalance)
  reader.require_int64("contexts_live_after", 0L, ResourceBalance)
  reader.require_int64("streams_live_after", 0L, ResourceBalance)
  reader.require_int64("events_live_after", 0L, ResourceBalance)
  reader.require_int64("graphs_live_after", 0L, ResourceBalance)
  reader.require_int64("graph_execs_live_after", 0L, ResourceBalance)
  reader.require_int64("allocations_live_after", 0L, ResourceBalance)
  reader.require_int64("modules_live_after", 0L, ResourceBalance)
  reader.require_int64("device_bytes_live_after", 0L, ResourceBalance)
  reader.require_int64("pending_work_after", 0L, ResourceBalance)
  reader.require_literal("manifest_bindable", "false", Authority)
  reader.require_literal("promotion_authority", "absent", Authority)
  reader.require_literal(
    "schema",
    "lunaflux-fused-approved-device-observation.v1",
    CanonicalDocument,
  )
  ignore(reader.read_value("device_uuid"))
  ignore(reader.read_value("device_pci"))
  ignore(reader.read_digest("device_name_sha256"))
  if reader.read_int64("device_total_memory_bytes") <= 0L {
    raise FusedPhysicalEvidenceMismatch(Target)
  }
  reader.require_literal("device_compute", "12.0", Target)
  ignore(reader.read_value("host_driver_version"))
  ignore(reader.read_digest("device_identity_record_sha256"))
  reader.require_digest(
    "approved_policy_sha256",
    expected_approved_policy_digest,
    Authority,
  )
  ignore(reader.read_digest("nvcc_sha256"))
  ignore(reader.read_digest("compute_sanitizer_sha256"))
  ignore(reader.read_digest("referee_source_sha256"))
  ignore(reader.read_digest("probe_executable_sha256"))
  reader.require_digest(
    "standalone_qkv_source_sha256",
    qkv.fallbacks[0].source,
    Referee,
  )
  reader.require_digest(
    "standalone_qkv_recipe_sha256",
    qkv.fallbacks[0].recipe,
    Referee,
  )
  ignore(reader.read_digest("standalone_qkv_cubin_sha256"))
  ignore(reader.read_digest("standalone_qkv_receipt_sha256"))
  reader.require_digest(
    "standalone_rope_source_sha256",
    qkv.fallbacks[1].source,
    Referee,
  )
  reader.require_digest(
    "standalone_rope_recipe_sha256",
    qkv.fallbacks[1].recipe,
    Referee,
  )
  ignore(reader.read_digest("standalone_rope_cubin_sha256"))
  ignore(reader.read_digest("standalone_rope_receipt_sha256"))
  reader.require_digest(
    "standalone_kv_write_source_sha256",
    qkv.fallbacks[2].source,
    Referee,
  )
  reader.require_digest(
    "standalone_kv_write_recipe_sha256",
    qkv.fallbacks[2].recipe,
    Referee,
  )
  ignore(reader.read_digest("standalone_kv_write_cubin_sha256"))
  ignore(reader.read_digest("standalone_kv_write_receipt_sha256"))
  ignore(reader.read_digest("standalone_oracle_output_sha256"))
  reader.require_literal("standalone_oracle_executed", "true", Referee)
  ignore(reader.read_digest("ptxas_sha256"))
  ignore(reader.read_digest("cuobjdump_sha256"))
  ignore(reader.read_digest("nvdisasm_sha256"))
  ignore(reader.read_digest("ptxas_version_sha256"))
  ignore(reader.read_digest("cuobjdump_version_sha256"))
  ignore(reader.read_digest("nvdisasm_version_sha256"))
  fused_physical_admit_resources(reader, "qkv_", 0)
  fused_physical_admit_resources(reader, "residual_", 512)
  (referee_digest, artifact_seal, evidence_seal)
}

///|
fn fused_physical_admit_resources(
  reader : FusedPhysicalReader,
  prefix : String,
  expected_static_shared : Int,
) -> Unit raise FusedPhysicalEvidenceError {
  ignore(reader.read_digest(prefix + "cuobjdump_sass_sha256"))
  let cu_count = reader.read_int(prefix + "cuobjdump_sass_instruction_count")
  ignore(reader.read_digest(prefix + "nvdisasm_sass_sha256"))
  let nv_count = reader.read_int(prefix + "nvdisasm_sass_instruction_count")
  let global_loads = reader.read_int(prefix + "global_load_instruction_count")
  let global_stores = reader.read_int(prefix + "global_store_instruction_count")
  let registers = reader.read_int(prefix + "registers_per_thread")
  let static_shared = reader.read_int(prefix + "static_shared_bytes")
  let local_bytes = reader.read_int(prefix + "local_bytes")
  let stack = reader.read_int(prefix + "stack_bytes")
  let spill_store = reader.read_int(prefix + "spill_store_bytes")
  let spill_load = reader.read_int(prefix + "spill_load_bytes")
  if cu_count <= 0 ||
    cu_count != nv_count ||
    global_loads <= 0 ||
    global_stores <= 0 ||
    registers <= 0 ||
    registers > 128 ||
    static_shared != expected_static_shared ||
    local_bytes != 0 ||
    stack != 0 ||
    spill_store != 0 ||
    spill_load != 0 {
    raise FusedPhysicalEvidenceMismatch(ResourceBalance)
  }
}

///|
/// Admits one exact, digest-pinned physical qualification record for the two
/// fused Phase-5 candidates and their already authenticated deterministic
/// compiler outputs.
///
/// The canonical record is parsed positionally. Duplicate, unknown, missing,
/// or reordered fields therefore fail before publication. The result remains
/// observation-only and cannot be converted into a runtime manifest or
/// promotion capability.
pub fn admit_fused_physical_evidence(
  qkv_candidate : @luna_cuda_fused_parallel_aot.FusedQkvRopeKvWriteCudaCandidate,
  qkv_compiled : @luna_cuda_fused_parallel_aot.FusedParallelCompiledArtifactBinding,
  residual_candidate : @luna_cuda_fused_parallel_aot.FusedResidualRmsNormCudaCandidate,
  residual_compiled : @luna_cuda_fused_parallel_aot.FusedParallelCompiledArtifactBinding,
  canonical : Bytes,
  expected_digest~ : @luna_cuda_fused_parallel_aot.FusedParallelDigest,
  expected_approved_policy_digest~ : @luna_cuda_fused_parallel_aot.FusedParallelDigest,
) -> AdmittedFusedPhysicalEvidence raise FusedPhysicalEvidenceError {
  let reader = fused_physical_reader(canonical)
  let digest = fused_physical_digest(canonical)
  if digest != expected_digest {
    raise FusedPhysicalEvidenceMismatch(Digest)
  }
  let qkv = fused_physical_expected_qkv(qkv_candidate, qkv_compiled)
  let residual = fused_physical_expected_residual(
    residual_candidate, residual_compiled,
  )
  fused_physical_validate_expected_family(qkv)
  fused_physical_validate_expected_family(residual)
  fused_physical_validate_expected_pair(qkv, residual)
  let (
    referee_shape_set_digest,
    artifact_directory_seal_digest,
    evidence_directory_seal_digest,
  ) = fused_physical_admit_global(
    reader, qkv, residual, expected_approved_policy_digest,
  )
  let admitted_qkv = fused_physical_admit_family(reader, "qkv_", qkv)
  let admitted_residual = fused_physical_admit_family(
    reader, "residual_", residual,
  )
  reader.finish()
  {
    canonical,
    digest,
    referee_shape_set_digest,
    artifact_directory_seal_digest,
    evidence_directory_seal_digest,
    approved_policy_digest: expected_approved_policy_digest,
    qkv: admitted_qkv,
    residual_norm: admitted_residual,
  }
}
