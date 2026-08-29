///|
pub(all) enum PagedAttentionReadOnlyPhysicalEvidenceRule {
  Canonical
  Digest
  Candidate
  CompiledBinding
  Fallback
  Target
  Layout
  Toolchain
  Sass
  Resource
  Numerical
  CacheImmutability
  Canary
  Sanitizer
  Seal
  Authority
} derive(Debug, Eq)

///|
pub(all) suberror PagedAttentionReadOnlyPhysicalEvidenceError {
  PagedAttentionReadOnlyPhysicalEvidenceInvalid(
    PagedAttentionReadOnlyPhysicalEvidenceRule
  )
  PagedAttentionReadOnlyPhysicalEvidenceMismatch(
    PagedAttentionReadOnlyPhysicalEvidenceRule
  )
  PagedAttentionReadOnlyPhysicalEvidenceLimitExceeded
} derive(Debug, Eq)
