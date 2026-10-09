///|
pub(all) enum CudaOfflineCompileDigestKind {
  Source
  Recipe
  Driver
  Receipt
} derive(Debug, Eq)

///|
pub(all) enum CudaOfflineCompileEvidenceStatus {
  InertSelfConsistent
} derive(Debug, Eq)

///|
pub struct CudaOfflineCompileEvidenceLimits {
  priv maximum_module_bytes : Int64
  priv maximum_receipt_bytes : Int
  priv maximum_symbol_bytes : Int
}

///|
pub struct CudaOfflineCompileEvidenceDigest(String) derive(Debug, Eq, Hash)

///|
pub struct CudaOfflineCompileEvidence {
  priv source_digest_sha256 : String
  priv recipe_digest_sha256 : String
  priv compiler : @luna_specialization.LunaCompilerPolicy
  priv driver_identity_sha256 : String
  priv target : @catalog.DeviceTarget
  priv function_symbol : String
  priv module_digest : @catalog.AotArtifactDigest
  priv module_byte_count : Int
  priv receipt_digest : CudaOfflineCompileEvidenceDigest
}

///|
pub(all) suberror CudaOfflineCompileEvidenceError {
  InvalidLimit
  InvalidDigest(CudaOfflineCompileDigestKind)
  InvalidFunctionSymbol
  EmptyModule
  ModuleTooLarge
  NondeterministicModule
  EmptyReceipt
  ReceiptTooLarge
  ReceiptDigestMismatch
  ReceiptContentMismatch
  InvalidModuleDigest
  ProducerAuthorityUnavailable
  CompilerExecutionProofUnavailable
  ExecutionAuthorityUnavailable
} derive(Debug, Eq)
