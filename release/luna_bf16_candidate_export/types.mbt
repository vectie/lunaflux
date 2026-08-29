///|
pub(all) enum LunaBf16CandidateExportStage {
  OutputPath
  CandidateSet
  Create
  Write
  Publish
  Cleanup
} derive(Debug, Eq)

///|
pub(all) suberror LunaBf16CandidateExportError {
  ExportFailed(LunaBf16CandidateExportStage)
  ExportFailedAndCleanup(
    LunaBf16CandidateExportStage,
    LunaBf16CandidateExportStage
  )
} derive(Debug, Eq)

///|
/// Root-free evidence for one atomically published candidate-set envelope.
pub struct LunaBf16CandidateExportEvidence {
  priv model_content_sha256 : String
  priv model_plan_sha256 : String
  priv target : String
  priv operation_count : Int
  priv candidate_set_sha256 : String
  priv inventory_sha256 : String
  priv export_record_sha256 : String
}

///|
pub fn LunaBf16CandidateExportEvidence::model_content_sha256(
  self : LunaBf16CandidateExportEvidence,
) -> String {
  self.model_content_sha256
}

///|
pub fn LunaBf16CandidateExportEvidence::model_plan_sha256(
  self : LunaBf16CandidateExportEvidence,
) -> String {
  self.model_plan_sha256
}

///|
pub fn LunaBf16CandidateExportEvidence::target(
  self : LunaBf16CandidateExportEvidence,
) -> String {
  self.target
}

///|
pub fn LunaBf16CandidateExportEvidence::operation_count(
  self : LunaBf16CandidateExportEvidence,
) -> Int {
  self.operation_count
}

///|
pub fn LunaBf16CandidateExportEvidence::candidate_set_sha256(
  self : LunaBf16CandidateExportEvidence,
) -> String {
  self.candidate_set_sha256
}

///|
pub fn LunaBf16CandidateExportEvidence::inventory_sha256(
  self : LunaBf16CandidateExportEvidence,
) -> String {
  self.inventory_sha256
}

///|
pub fn LunaBf16CandidateExportEvidence::export_record_sha256(
  self : LunaBf16CandidateExportEvidence,
) -> String {
  self.export_record_sha256
}

///|
pub fn LunaBf16CandidateExportEvidence::candidate_root_relative(
  _self : LunaBf16CandidateExportEvidence,
) -> String {
  "candidate-root"
}

///|
pub fn LunaBf16CandidateExportEvidence::inventory_relative(
  _self : LunaBf16CandidateExportEvidence,
) -> String {
  "candidate.files.sha256"
}
