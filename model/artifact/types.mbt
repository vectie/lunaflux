///|
/// The three inert files admitted by the dense-Llama v1 bundle loader.
pub(all) enum ArtifactFile {
  ModelConfig
  Tokenizer
  Weights
} derive(Debug, Eq)

///|
/// Filesystem operations whose failures remain payload-safe.
pub(all) enum ArtifactFileOperation {
  OpenFile
  StatFile
  SnapshotFile
  CloseFile
} derive(Debug, Eq)

///|
/// Bounded filesystem failure classes with no path or native error text.
pub(all) enum ArtifactFileIoIssue {
  AuthorityClosed
  Busy
  Failed
} derive(Debug, Eq)

///|
/// Resource-bound categories for bundle admission.
pub(all) enum ArtifactLimit {
  ModelConfigBytes
  TokenizerBytes
  WeightsBytes
  TotalBytes
  ModelConfigDepth
  TokenizerDepth
} derive(Debug, Eq)

///|
/// Typed, payload-safe bundle admission failures.
///
/// No variant retains a filesystem path, file contents, or vendor error text.
pub(all) suberror ArtifactError {
  InvalidDigest
  InvalidLocator(ArtifactFile)
  DuplicateLocator(ArtifactFile)
  InvalidLimit(ArtifactLimit)
  FileUnavailable(ArtifactFile)
  UnsupportedPathType(ArtifactFile)
  FileIo(
    file~ : ArtifactFile,
    operation~ : ArtifactFileOperation,
    issue~ : ArtifactFileIoIssue
  )
  FileTooLarge(file~ : ArtifactFile, maximum~ : Int64)
  TotalTooLarge(maximum~ : Int64)
  FileTruncated(ArtifactFile)
  FileChangedDuringRead(ArtifactFile)
  DigestMismatch(ArtifactFile)
  TokenizerVocabularyMismatch
  InvalidModelConfig(@config_reader.ModelConfigReaderError)
  InvalidTokenizer(@json_reader.TokenizerJsonError)
  InvalidSafetensors(@safetensors_reader.SafetensorsReaderError)
} derive(Debug, Eq)

///|
/// Opaque canonical lowercase SHA-256 identity for an admitted file.
pub struct ArtifactDigest {
  hex : String
} derive(Debug, Eq)

///|
/// Opaque approved relative source for the three v1 bundle files.
///
/// LunaFlux does not scan the approved root or infer filenames. Locator text
/// remains private to the filesystem capability boundary.
pub struct ArtifactSource {
  priv model_config : @approved_fs.ApprovedRelativeLocator
  priv tokenizer : @approved_fs.ApprovedRelativeLocator
  priv weights : @approved_fs.ApprovedRelativeLocator
}

///|
/// Independently approved file identities supplied before admission.
pub struct ExpectedDigests {
  model_config : ArtifactDigest
  tokenizer : ArtifactDigest
  weights : ArtifactDigest
} derive(Debug, Eq)

///|
/// Resource limits applied before each file allocation.
pub struct ArtifactLimits {
  max_model_config_bytes : Int64
  max_tokenizer_bytes : Int64
  max_weights_bytes : Int64
  max_total_bytes : Int64
  max_model_config_depth : Int
  max_tokenizer_depth : Int
  tokenizer_limits : @tokenizer.TokenizerLimits
  safetensors_limits : @safetensors.HeaderLimits
} derive(Debug, Eq)

///|
/// Validated identities of the exact bytes retained by an admitted bundle.
pub struct ArtifactIdentity {
  model_config : ArtifactDigest
  tokenizer : @tokenizer.TokenizerDigest
  model : @spec.ModelIdentity
} derive(Debug, Eq)

///|
/// Immutable Phase 1 host/reference snapshot of one admitted v1 bundle.
///
/// This whole-file representation is deliberately not the production loader;
/// production weights must stream or map directly to final destinations.
pub struct ReferenceArtifactBundle {
  model_config_bytes : Bytes
  tokenizer_bytes : Bytes
  weights_bytes : Bytes
  model : @spec.LlamaModelMetadata
  tokenizer : @tokenizer.TokenizerSpec
  safetensors : @safetensors.SafetensorsMetadata
  identity : ArtifactIdentity
}
