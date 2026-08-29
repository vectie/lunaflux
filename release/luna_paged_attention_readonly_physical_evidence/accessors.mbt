///|
pub fn AdmittedPagedAttentionReadOnlyPhysicalEvidence::candidate_digest(
  self : AdmittedPagedAttentionReadOnlyPhysicalEvidence,
) -> String {
  self.candidate_digest
}

///|
pub fn AdmittedPagedAttentionReadOnlyPhysicalEvidence::compiled_binding_digest(
  self : AdmittedPagedAttentionReadOnlyPhysicalEvidence,
) -> String {
  self.compiled_binding_digest
}

///|
pub fn AdmittedPagedAttentionReadOnlyPhysicalEvidence::cubin_digest(
  self : AdmittedPagedAttentionReadOnlyPhysicalEvidence,
) -> PagedAttentionReadOnlyPhysicalEvidenceDigest {
  self.cubin_digest
}

///|
pub fn AdmittedPagedAttentionReadOnlyPhysicalEvidence::operation_id(
  self : AdmittedPagedAttentionReadOnlyPhysicalEvidence,
) -> @plan.OperationId {
  self.operation_id
}

///|
pub fn AdmittedPagedAttentionReadOnlyPhysicalEvidence::input_activation(
  self : AdmittedPagedAttentionReadOnlyPhysicalEvidence,
) -> @plan.ActivationRef {
  self.input_activation
}

///|
pub fn AdmittedPagedAttentionReadOnlyPhysicalEvidence::input_row_width(
  self : AdmittedPagedAttentionReadOnlyPhysicalEvidence,
) -> Int {
  self.input_row_width
}

///|
pub fn AdmittedPagedAttentionReadOnlyPhysicalEvidence::dispatch_canary_per_token(
  self : AdmittedPagedAttentionReadOnlyPhysicalEvidence,
) -> Int {
  self.dispatch_canary_per_token
}

///|
pub fn AdmittedPagedAttentionReadOnlyPhysicalEvidence::dispatch_canary_cell_count(
  self : AdmittedPagedAttentionReadOnlyPhysicalEvidence,
) -> Int {
  self.dispatch_canary_cell_count
}

///|
pub fn AdmittedPagedAttentionReadOnlyPhysicalEvidence::numeric_case_count(
  self : AdmittedPagedAttentionReadOnlyPhysicalEvidence,
) -> Int {
  self.numeric_case_count
}

///|
pub fn AdmittedPagedAttentionReadOnlyPhysicalEvidence::canonical_bytes(
  self : AdmittedPagedAttentionReadOnlyPhysicalEvidence,
) -> Bytes {
  self.canonical
}

///|
pub fn AdmittedPagedAttentionReadOnlyPhysicalEvidence::digest(
  self : AdmittedPagedAttentionReadOnlyPhysicalEvidence,
) -> PagedAttentionReadOnlyPhysicalEvidenceDigest {
  self.digest
}

///|
pub fn AdmittedPagedAttentionReadOnlyPhysicalEvidence::files_manifest_digest(
  self : AdmittedPagedAttentionReadOnlyPhysicalEvidence,
) -> PagedAttentionReadOnlyPhysicalEvidenceDigest {
  self.files_manifest_digest
}

///|
pub fn AdmittedPagedAttentionReadOnlyPhysicalEvidence::outer_seal_digest(
  self : AdmittedPagedAttentionReadOnlyPhysicalEvidence,
) -> PagedAttentionReadOnlyPhysicalEvidenceDigest {
  self.outer_seal_digest
}

///|
pub fn AdmittedPagedAttentionReadOnlyPhysicalEvidence::manifest_bindable(
  _self : AdmittedPagedAttentionReadOnlyPhysicalEvidence,
) -> Bool {
  false
}

///|
pub fn AdmittedPagedAttentionReadOnlyPhysicalEvidence::promotion_authorized(
  _self : AdmittedPagedAttentionReadOnlyPhysicalEvidence,
) -> Bool {
  false
}

///|
pub fn QualifiedPagedAttentionReadOnlyArtifact::candidate(
  self : QualifiedPagedAttentionReadOnlyArtifact,
) -> @luna_cuda_paged_attention_readonly_aot.PagedAttentionReadOnlyCudaCandidate {
  self.candidate
}

///|
pub fn QualifiedPagedAttentionReadOnlyArtifact::compiled(
  self : QualifiedPagedAttentionReadOnlyArtifact,
) -> @luna_cuda_paged_attention_readonly_aot.PagedAttentionReadOnlyCompiledBinding {
  self.compiled
}

///|
pub fn QualifiedPagedAttentionReadOnlyArtifact::evidence(
  self : QualifiedPagedAttentionReadOnlyArtifact,
) -> AdmittedPagedAttentionReadOnlyPhysicalEvidence {
  self.evidence
}

///|
pub fn QualifiedPagedAttentionReadOnlyArtifact::physically_qualified(
  _self : QualifiedPagedAttentionReadOnlyArtifact,
) -> Bool {
  true
}

///|
pub fn QualifiedPagedAttentionReadOnlyArtifact::manifest_bindable(
  _self : QualifiedPagedAttentionReadOnlyArtifact,
) -> Bool {
  false
}

///|
pub fn QualifiedPagedAttentionReadOnlyArtifact::promotion_authorized(
  _self : QualifiedPagedAttentionReadOnlyArtifact,
) -> Bool {
  false
}
