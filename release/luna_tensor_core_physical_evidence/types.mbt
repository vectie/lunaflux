///|
const TENSOR_CORE_EVIDENCE_RESULT_LINES : Int = 67

///|
const TENSOR_CORE_EVIDENCE_MAXIMUM_RESULT_BYTES : Int = 65536

///|
const TENSOR_CORE_EVIDENCE_MAXIMUM_MANIFEST_BYTES : Int = 262144

///|
priv struct LunaTensorCoreToolFacts {
  driver_identity : LunaTensorCorePhysicalEvidenceDigest
  driver_record : LunaTensorCorePhysicalEvidenceDigest
  nvcc_version : LunaTensorCorePhysicalEvidenceDigest
  ptxas_version : LunaTensorCorePhysicalEvidenceDigest
  cuobjdump_version : LunaTensorCorePhysicalEvidenceDigest
  nvdisasm_version : LunaTensorCorePhysicalEvidenceDigest
  sanitizer_version : LunaTensorCorePhysicalEvidenceDigest
  instruction_family : String
  cuobjdump_sass : LunaTensorCorePhysicalEvidenceDigest
  nvdisasm_sass : LunaTensorCorePhysicalEvidenceDigest
}

///|
pub struct AdmittedLunaTensorCorePhysicalEvidence {
  priv candidate_digest : String
  priv source_digest : String
  priv serial_oracle_digest : @luna_tile_ir.LunaTileCudaTranslationDigest
  priv parallel_fallback_digest : @luna_tile_ir.LunaTileParallelDigest
  priv cubin_digest : LunaTensorCorePhysicalEvidenceDigest
  priv driver_identity_digest : LunaTensorCorePhysicalEvidenceDigest
  priv instruction_family : String
  priv numeric_case_count : Int
  priv registers_per_thread : Int
  priv canonical : Bytes
  priv digest : LunaTensorCorePhysicalEvidenceDigest
  priv files_manifest_digest : LunaTensorCorePhysicalEvidenceDigest
  priv outer_seal_digest : LunaTensorCorePhysicalEvidenceDigest
}

///|
pub struct QualifiedLunaTileBf16Mma16x16x16Artifact {
  priv candidate : @luna_tensor_core_candidate.LunaTileBf16Mma16x16x16Candidate
  priv evidence : AdmittedLunaTensorCorePhysicalEvidence
}

///|
pub struct LunaTensorCorePhysicalEvidenceLimits {
  priv maximum_result_bytes : Int
  priv maximum_manifest_bytes : Int
}

///|
pub fn LunaTensorCorePhysicalEvidenceLimits::new(
  maximum_result_bytes~ : Int,
  maximum_manifest_bytes~ : Int,
) -> LunaTensorCorePhysicalEvidenceLimits raise LunaTensorCorePhysicalEvidenceError {
  if maximum_result_bytes <= 0 ||
    maximum_result_bytes > TENSOR_CORE_EVIDENCE_MAXIMUM_RESULT_BYTES ||
    maximum_manifest_bytes <= 0 ||
    maximum_manifest_bytes > TENSOR_CORE_EVIDENCE_MAXIMUM_MANIFEST_BYTES {
    raise LunaTensorCorePhysicalEvidenceLimitExceeded
  }
  { maximum_result_bytes, maximum_manifest_bytes }
}
