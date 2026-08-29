///|
pub fn AdmittedLunaTileSimtPhysicalEvidence::source_digest(
  self : AdmittedLunaTileSimtPhysicalEvidence,
) -> @luna_tile_ir.LunaTileCudaSourceDigest {
  self.source
}

///|
pub fn AdmittedLunaTileSimtPhysicalEvidence::serial_module_digest(
  self : AdmittedLunaTileSimtPhysicalEvidence,
) -> LunaTileOptimizerDigest {
  self.serial_module_digest
}

///|
pub fn AdmittedLunaTileSimtPhysicalEvidence::sealed_result_digest(
  self : AdmittedLunaTileSimtPhysicalEvidence,
) -> LunaTileOptimizerDigest {
  self.sealed_result
}

///|
pub fn AdmittedLunaTileSimtPhysicalEvidence::canonical_bytes(
  self : AdmittedLunaTileSimtPhysicalEvidence,
) -> Bytes {
  self.canonical
}

///|
pub fn AdmittedLunaTilePairedBenchmarkWin::observation_count(
  self : AdmittedLunaTilePairedBenchmarkWin,
) -> Int {
  self.observation_count
}

///|
pub fn AdmittedLunaTilePairedBenchmarkWin::candidate_total_ns(
  self : AdmittedLunaTilePairedBenchmarkWin,
) -> Int64 {
  self.candidate_total_ns
}

///|
pub fn AdmittedLunaTilePairedBenchmarkWin::fallback_total_ns(
  self : AdmittedLunaTilePairedBenchmarkWin,
) -> Int64 {
  self.fallback_total_ns
}

///|
pub fn AdmittedLunaTilePairedBenchmarkWin::canonical_bytes(
  self : AdmittedLunaTilePairedBenchmarkWin,
) -> Bytes {
  self.canonical
}

///|
pub fn LunaTilePromotionApprovalRecord::canonical_bytes(
  self : LunaTilePromotionApprovalRecord,
) -> Bytes {
  self.canonical
}

///|
pub fn LunaTilePromotionApprovalRecord::path(
  self : LunaTilePromotionApprovalRecord,
) -> LunaTileOptimizerPath {
  self.path
}

///|
pub fn LunaTilePromotionApprovalRecord::source_digest(
  self : LunaTilePromotionApprovalRecord,
) -> LunaTileOptimizerDigest {
  self.source
}

///|
pub fn LunaTilePromotionApprovalRecord::entry_point(
  self : LunaTilePromotionApprovalRecord,
) -> String {
  self.entry_point
}

///|
pub fn LunaTilePromotionApprovalRecord::physical_evidence_digest(
  self : LunaTilePromotionApprovalRecord,
) -> LunaTileOptimizerDigest {
  self.physical
}

///|
pub fn LunaTilePromotionApprovalRecord::benchmark_digest(
  self : LunaTilePromotionApprovalRecord,
) -> LunaTileOptimizerDigest {
  self.benchmark
}

///|
pub fn LunaTilePromotionApprovalRecord::digest(
  self : LunaTilePromotionApprovalRecord,
) -> LunaTileOptimizerDigest {
  self.digest
}

///|
pub fn LunaTilePromotionApprovalRecord::candidate_digest(
  self : LunaTilePromotionApprovalRecord,
) -> LunaTileOptimizerDigest {
  self.candidate
}

///|
pub fn LunaTilePromotionApprovalRecord::fallback_digest(
  self : LunaTilePromotionApprovalRecord,
) -> LunaTileOptimizerDigest {
  self.fallback
}

///|
pub fn ApprovedLunaTileOptimizerCatalogEntry::digest(
  self : ApprovedLunaTileOptimizerCatalogEntry,
) -> LunaTileOptimizerDigest {
  self.record.digest
}

///|
pub fn ApprovedLunaTileOptimizerCatalogEntry::path(
  self : ApprovedLunaTileOptimizerCatalogEntry,
) -> LunaTileOptimizerPath {
  self.record.path
}

///|
pub fn ApprovedLunaTileOptimizerCatalogEntry::candidate_digest(
  self : ApprovedLunaTileOptimizerCatalogEntry,
) -> LunaTileOptimizerDigest {
  self.record.candidate
}

///|
pub fn ApprovedLunaTileOptimizerCatalogEntry::source_digest(
  self : ApprovedLunaTileOptimizerCatalogEntry,
) -> LunaTileOptimizerDigest {
  self.record.source
}

///|
pub fn ApprovedLunaTileOptimizerCatalogEntry::entry_point(
  self : ApprovedLunaTileOptimizerCatalogEntry,
) -> String {
  self.record.entry_point
}

///|
pub fn ApprovedLunaTileOptimizerCatalogEntry::deployment_identity(
  self : ApprovedLunaTileOptimizerCatalogEntry,
) -> LunaTilePromotionDeploymentIdentity {
  self.record.deployment
}

///|
pub fn ApprovedLunaTileOptimizerCatalogEntry::promotion_epoch(
  self : ApprovedLunaTileOptimizerCatalogEntry,
) -> UInt64 {
  self.record.promotion_epoch
}

///|
pub fn ApprovedLunaTileOptimizerCatalogEntry::physical_evidence_digest(
  self : ApprovedLunaTileOptimizerCatalogEntry,
) -> LunaTileOptimizerDigest {
  self.record.physical
}

///|
pub fn ApprovedLunaTileOptimizerCatalogEntry::benchmark_digest(
  self : ApprovedLunaTileOptimizerCatalogEntry,
) -> LunaTileOptimizerDigest {
  self.record.benchmark
}

///|
pub fn ApprovedLunaTileOptimizerCatalogEntry::module_digest(
  self : ApprovedLunaTileOptimizerCatalogEntry,
) -> LunaTileOptimizerDigest {
  self.record.module_digest
}

///|
pub fn ApprovedLunaTileOptimizerCatalogEntry::fallback_digest(
  self : ApprovedLunaTileOptimizerCatalogEntry,
) -> LunaTileOptimizerDigest {
  self.record.fallback
}

///|
pub fn ApprovedLunaTileOptimizerCatalogEntry::external_approval_record(
  self : ApprovedLunaTileOptimizerCatalogEntry,
) -> @luna_capability_manifest.LunaAdmittedExternalRecord {
  self.approval
}

///|
pub fn ApprovedLunaTileOptimizerCatalogEntry::manifest_bindable(
  _self : ApprovedLunaTileOptimizerCatalogEntry,
) -> Bool {
  true
}

///|
pub fn ApprovedLunaTileOptimizerCatalogEntry::startup_selectable(
  _self : ApprovedLunaTileOptimizerCatalogEntry,
) -> Bool {
  true
}
