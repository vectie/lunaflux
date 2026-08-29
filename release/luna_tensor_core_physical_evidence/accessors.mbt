///|
pub fn AdmittedLunaTensorCorePhysicalEvidence::candidate_digest(
  self : AdmittedLunaTensorCorePhysicalEvidence,
) -> String {
  self.candidate_digest
}

///|
pub fn AdmittedLunaTensorCorePhysicalEvidence::source_digest(
  self : AdmittedLunaTensorCorePhysicalEvidence,
) -> String {
  self.source_digest
}

///|
pub fn AdmittedLunaTensorCorePhysicalEvidence::serial_oracle_digest(
  self : AdmittedLunaTensorCorePhysicalEvidence,
) -> @luna_tile_ir.LunaTileCudaTranslationDigest {
  self.serial_oracle_digest
}

///|
pub fn AdmittedLunaTensorCorePhysicalEvidence::parallel_fallback_digest(
  self : AdmittedLunaTensorCorePhysicalEvidence,
) -> @luna_tile_ir.LunaTileParallelDigest {
  self.parallel_fallback_digest
}

///|
pub fn AdmittedLunaTensorCorePhysicalEvidence::cubin_digest(
  self : AdmittedLunaTensorCorePhysicalEvidence,
) -> LunaTensorCorePhysicalEvidenceDigest {
  self.cubin_digest
}

///|
pub fn AdmittedLunaTensorCorePhysicalEvidence::driver_identity_digest(
  self : AdmittedLunaTensorCorePhysicalEvidence,
) -> LunaTensorCorePhysicalEvidenceDigest {
  self.driver_identity_digest
}

///|
pub fn AdmittedLunaTensorCorePhysicalEvidence::instruction_family(
  self : AdmittedLunaTensorCorePhysicalEvidence,
) -> String {
  self.instruction_family
}

///|
pub fn AdmittedLunaTensorCorePhysicalEvidence::numeric_case_count(
  self : AdmittedLunaTensorCorePhysicalEvidence,
) -> Int {
  self.numeric_case_count
}

///|
pub fn AdmittedLunaTensorCorePhysicalEvidence::registers_per_thread(
  self : AdmittedLunaTensorCorePhysicalEvidence,
) -> Int {
  self.registers_per_thread
}

///|
pub fn AdmittedLunaTensorCorePhysicalEvidence::canonical_bytes(
  self : AdmittedLunaTensorCorePhysicalEvidence,
) -> Bytes {
  self.canonical
}

///|
pub fn AdmittedLunaTensorCorePhysicalEvidence::digest(
  self : AdmittedLunaTensorCorePhysicalEvidence,
) -> LunaTensorCorePhysicalEvidenceDigest {
  self.digest
}

///|
pub fn AdmittedLunaTensorCorePhysicalEvidence::files_manifest_digest(
  self : AdmittedLunaTensorCorePhysicalEvidence,
) -> LunaTensorCorePhysicalEvidenceDigest {
  self.files_manifest_digest
}

///|
pub fn AdmittedLunaTensorCorePhysicalEvidence::outer_seal_digest(
  self : AdmittedLunaTensorCorePhysicalEvidence,
) -> LunaTensorCorePhysicalEvidenceDigest {
  self.outer_seal_digest
}

///|
pub fn AdmittedLunaTensorCorePhysicalEvidence::manifest_bindable(
  _self : AdmittedLunaTensorCorePhysicalEvidence,
) -> Bool {
  false
}

///|
pub fn AdmittedLunaTensorCorePhysicalEvidence::promotion_authorized(
  _self : AdmittedLunaTensorCorePhysicalEvidence,
) -> Bool {
  false
}

///|
pub fn QualifiedLunaTileBf16Mma16x16x16Artifact::candidate(
  self : QualifiedLunaTileBf16Mma16x16x16Artifact,
) -> @luna_tensor_core_candidate.LunaTileBf16Mma16x16x16Candidate {
  self.candidate
}

///|
pub fn QualifiedLunaTileBf16Mma16x16x16Artifact::evidence(
  self : QualifiedLunaTileBf16Mma16x16x16Artifact,
) -> AdmittedLunaTensorCorePhysicalEvidence {
  self.evidence
}

///|
pub fn QualifiedLunaTileBf16Mma16x16x16Artifact::physically_qualified(
  _self : QualifiedLunaTileBf16Mma16x16x16Artifact,
) -> Bool {
  true
}

///|
pub fn QualifiedLunaTileBf16Mma16x16x16Artifact::manifest_bindable(
  _self : QualifiedLunaTileBf16Mma16x16x16Artifact,
) -> Bool {
  false
}

///|
pub fn QualifiedLunaTileBf16Mma16x16x16Artifact::promotion_authorized(
  _self : QualifiedLunaTileBf16Mma16x16x16Artifact,
) -> Bool {
  false
}
