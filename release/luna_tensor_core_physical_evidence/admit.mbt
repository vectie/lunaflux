///|
fn tensor_core_admit_identity(
  reader : LunaTensorCoreEvidenceReader,
  candidate : @luna_tensor_core_candidate.LunaTileBf16Mma16x16x16Candidate,
  serial : @luna_tile_ir.LunaTileCudaTranslationUnit,
  parallel : @luna_tile_ir.LunaTileParallelCandidate,
) -> (
  LunaTensorCorePhysicalEvidenceDigest,
  LunaTensorCorePhysicalEvidenceDigest,
  LunaTensorCorePhysicalEvidenceDigest,
  LunaTensorCorePhysicalEvidenceDigest,
) raise LunaTensorCorePhysicalEvidenceError {
  reader.require_literal(
    "schema",
    "lunaflux-lunatile-tensor-core-physical-campaign.v1",
    CanonicalDocument,
  )
  reader.require_literal(
    "outcome",
    "lunatile-tensor-core-physical-campaign-pass",
    Target,
  )
  reader.require_digest_text(
    "program_sha256",
    candidate.program_digest().as_hex(),
    Candidate,
  )
  reader.require_digest_text(
    "serial_translation_sha256",
    serial.digest().as_hex(),
    Oracle,
  )
  reader.require_digest_text(
    "parallel_plan_sha256",
    parallel.digest().as_hex(),
    ParallelFallback,
  )
  reader.require_digest_text(
    "tensor_core_candidate_sha256",
    candidate.digest().as_hex(),
    Candidate,
  )
  let serial_source = reader.read_digest("serial_source_sha256")
  if serial_source.as_hex() != serial.source_digest().as_hex() {
    raise LunaTensorCorePhysicalEvidenceMismatch(Oracle)
  }
  let tensor_source = reader.read_digest("tensor_core_source_sha256")
  if tensor_source.as_hex() != candidate.source_digest().as_hex() {
    raise LunaTensorCorePhysicalEvidenceMismatch(Candidate)
  }
  let serial_cubin = reader.read_digest("serial_cubin_sha256")
  let tensor_cubin = reader.read_digest("tensor_core_cubin_sha256")
  reader.require_literal("target", "sm_120", Target)
  reader.require_int("device_ordinal", 0, Device)
  ignore(reader.read_digest("device_name_sha256"))
  let memory = reader.read_int64("device_total_memory_bytes")
  if memory <= 0L {
    raise LunaTensorCorePhysicalEvidenceMismatch(Device)
  }
  reader.require_literal("device_compute", "12.0", Device)
  reader.require_literal("device_bf16", "true", Device)
  (serial_source, tensor_source, serial_cubin, tensor_cubin)
}

///|
fn tensor_core_admit_tools(
  reader : LunaTensorCoreEvidenceReader,
) -> LunaTensorCoreToolFacts raise LunaTensorCorePhysicalEvidenceError {
  let driver_identity = reader.read_digest("driver_identity_sha256")
  let driver_record = reader.read_digest("driver_record_sha256")
  reader.require_literal("compiler_version", "13.1.115", Compiler)
  ignore(reader.read_digest("nvcc_sha256"))
  ignore(reader.read_digest("ptxas_sha256"))
  ignore(reader.read_digest("cuobjdump_sha256"))
  ignore(reader.read_digest("nvdisasm_sha256"))
  ignore(reader.read_digest("compute_sanitizer_sha256"))
  let nvcc_version = reader.read_digest("nvcc_version_sha256")
  let ptxas_version = reader.read_digest("ptxas_version_sha256")
  let cuobjdump_version = reader.read_digest("cuobjdump_version_sha256")
  let nvdisasm_version = reader.read_digest("nvdisasm_version_sha256")
  let sanitizer_version = reader.read_digest("compute_sanitizer_version_sha256")
  let instruction = reader.read_value("sass_instruction_family")
  if !(instruction == "HMMA" ||
    instruction == "MMA" ||
    instruction == "TCGEN05_MMA") {
    raise LunaTensorCorePhysicalEvidenceMismatch(Sass)
  }
  let cuobjdump_sass = reader.read_digest("cuobjdump_sass_sha256")
  let cuobjdump_count = reader.read_int("cuobjdump_sass_instruction_count")
  let nvdisasm_sass = reader.read_digest("nvdisasm_sass_sha256")
  let nvdisasm_count = reader.read_int("nvdisasm_sass_instruction_count")
  if cuobjdump_count <= 0 || cuobjdump_count != nvdisasm_count {
    raise LunaTensorCorePhysicalEvidenceMismatch(Sass)
  }
  reader.require_literal("sass_instruction_observed", "true", Sass)
  {
    driver_identity,
    driver_record,
    nvcc_version,
    ptxas_version,
    cuobjdump_version,
    nvdisasm_version,
    sanitizer_version,
    instruction_family: instruction,
    cuobjdump_sass,
    nvdisasm_sass,
  }
}

///|
fn tensor_core_admit_resources_and_numeric(
  reader : LunaTensorCoreEvidenceReader,
) -> (Int, Int) raise LunaTensorCorePhysicalEvidenceError {
  let registers = reader.read_int("registers_per_thread")
  reader.require_int("register_bound", 128, Resource)
  let static_shared = reader.read_int("static_shared_bytes")
  reader.require_int("static_shared_bound", 0, Resource)
  let stack = reader.read_int("stack_bytes")
  let local_bytes = reader.read_int("local_bytes")
  let spill_store = reader.read_int("spill_store_bytes")
  let spill_load = reader.read_int("spill_load_bytes")
  if registers <= 0 ||
    registers > 128 ||
    static_shared != 0 ||
    stack != 0 ||
    local_bytes != 0 ||
    spill_store != 0 ||
    spill_load != 0 {
    raise LunaTensorCorePhysicalEvidenceMismatch(Resource)
  }
  reader.require_literal("resource_bounds_passed", "true", Resource)
  let cases = reader.read_int("numeric_case_count")
  if cases < 16384 || cases > 524288 || cases % 16384 != 0 {
    raise LunaTensorCorePhysicalEvidenceMismatch(Numerical)
  }
  reader.require_literal("absolute_tolerance", "0.001", Numerical)
  reader.require_literal("relative_tolerance", "0.0001", Numerical)
  for
    key in [
      "cpu_vs_serial_max_abs_error", "cpu_vs_tensor_core_max_abs_error", "serial_vs_tensor_core_max_abs_error",
    ] {
    if reader.read_nonnegative_double(key) > 0.001 {
      raise LunaTensorCorePhysicalEvidenceMismatch(Numerical)
    }
  }
  reader.require_literal("cpu_oracle", "independent-ordered-f32-v1", Numerical)
  reader.require_literal("serial_cuda_oracle", "passed", Numerical)
  (registers, cases)
}

///|
fn tensor_core_admit_sanitizers(
  reader : LunaTensorCoreEvidenceReader,
) -> (
  LunaTensorCorePhysicalEvidenceDigest,
  LunaTensorCorePhysicalEvidenceDigest,
  LunaTensorCorePhysicalEvidenceDigest,
) raise LunaTensorCorePhysicalEvidenceError {
  let memcheck = reader.read_digest("memcheck_log_sha256")
  reader.require_int("memcheck_error_summary_count", 1, Sanitizer)
  reader.require_int("memcheck_errors", 0, Sanitizer)
  let racecheck = reader.read_digest("racecheck_log_sha256")
  reader.require_int("racecheck_error_summary_count", 1, Sanitizer)
  reader.require_int("racecheck_errors", 0, Sanitizer)
  let initcheck = reader.read_digest("initcheck_log_sha256")
  reader.require_int("initcheck_error_summary_count", 1, Sanitizer)
  reader.require_int("initcheck_errors", 0, Sanitizer)
  reader.require_literal(
    "cleanup_balance",
    "context0,stream0,allocation0,module0,function0,device_bytes0,pending0,cleanup0",
    Resource,
  )
  reader.require_literal("physical_cuda_observed", "true", Device)
  reader.require_literal("qualification_only", "true", Authority)
  reader.require_literal("manifest_bindable", "false", Authority)
  reader.require_literal("promotion_authority", "absent", Authority)
  (memcheck, racecheck, initcheck)
}

///|
pub fn admit_luna_tile_tensor_core_physical_evidence(
  candidate : @luna_tensor_core_candidate.LunaTileBf16Mma16x16x16Candidate,
  serial : @luna_tile_ir.LunaTileCudaTranslationUnit,
  parallel : @luna_tile_ir.LunaTileParallelCandidate,
  result : Bytes,
  files_manifest : Bytes,
  outer_seal : Bytes,
  expected_outer_seal_digest : LunaTensorCorePhysicalEvidenceDigest,
  limits : LunaTensorCorePhysicalEvidenceLimits,
) -> AdmittedLunaTensorCorePhysicalEvidence raise LunaTensorCorePhysicalEvidenceError {
  if candidate.serial_oracle_digest() != serial.digest() ||
    candidate.parallel_plan_digest() != parallel.digest() ||
    parallel.serial_oracle_digest() != serial.digest() ||
    parallel.compute_capability() != (12, 0) ||
    candidate.manifest_bindable() ||
    candidate.promotion_authorized() {
    raise LunaTensorCorePhysicalEvidenceMismatch(Candidate)
  }
  let (manifest_digest, result_digest) = tensor_core_require_outer_seal(
    result, files_manifest, outer_seal, expected_outer_seal_digest,
  )
  let reader = tensor_core_evidence_reader(result, limits.maximum_result_bytes)
  let (serial_source, tensor_source, serial_cubin, tensor_cubin) = tensor_core_admit_identity(
    reader, candidate, serial, parallel,
  )
  let tools = tensor_core_admit_tools(reader)
  let (registers, cases) = tensor_core_admit_resources_and_numeric(reader)
  let (memcheck, racecheck, initcheck) = tensor_core_admit_sanitizers(reader)
  let recorded_manifest = reader.read_digest("evidence_files_manifest_sha256")
  if recorded_manifest != manifest_digest {
    raise LunaTensorCorePhysicalEvidenceMismatch(Seal)
  }
  reader.finish()
  let entries = tensor_core_evidence_manifest_entries(
    files_manifest,
    limits.maximum_manifest_bytes,
  )
  tensor_core_require_critical_manifest(
    entries,
    tensor_core_evidence_read_digest(serial.digest().as_hex()),
    tensor_core_evidence_read_digest(candidate.digest().as_hex()),
    serial_source,
    tensor_source,
    serial_cubin,
    tensor_cubin,
    tools.driver_record,
    tools.nvcc_version,
    tools.ptxas_version,
    tools.cuobjdump_version,
    tools.nvdisasm_version,
    tools.sanitizer_version,
    tools.cuobjdump_sass,
    tools.nvdisasm_sass,
    memcheck,
    racecheck,
    initcheck,
  )
  {
    candidate_digest: candidate.digest().as_hex(),
    source_digest: candidate.source_digest().as_hex(),
    serial_oracle_digest: serial.digest(),
    parallel_fallback_digest: parallel.digest(),
    cubin_digest: tensor_cubin,
    driver_identity_digest: tools.driver_identity,
    instruction_family: tools.instruction_family,
    numeric_case_count: cases,
    registers_per_thread: registers,
    canonical: result,
    digest: result_digest,
    files_manifest_digest: manifest_digest,
    outer_seal_digest: expected_outer_seal_digest,
  }
}
