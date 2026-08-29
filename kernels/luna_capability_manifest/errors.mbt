///|
pub(all) enum LunaCapabilityManifestRule {
  Version
  Digest
  Architecture
  DType
  Layout
  ShapeClass
  Alignment
  Workspace
  GraphMemory
  EagerFallback
  Artifact
  Specialization
  Compiler
  NumericalPolicy
  Evidence
  SignatureAlgorithm
  SignatureEnvelope
  ExternalApproval
  ApprovedSource
  CanonicalBytes
} derive(Debug, Eq)

///|
pub(all) suberror LunaCapabilityManifestError {
  Invalid(LunaCapabilityManifestRule)
  Mismatch(LunaCapabilityManifestRule)
  Unsupported(LunaCapabilityManifestRule)
  SizeOverflow(LunaCapabilityManifestRule)
} derive(Debug, Eq)
