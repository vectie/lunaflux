///|
/// The three inert files admitted by the dense-Llama v1 bundle loader.
pub(all) enum ArtifactFile {
  ModelConfig
  Tokenizer
  Weights
} derive(Debug, Eq)

///|
/// Path validation categories that never retain path text.
pub(all) enum ArtifactPathIssue {
  Empty
  TooLong
  Duplicate
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
  InvalidPath(file~ : ArtifactFile, issue~ : ArtifactPathIssue)
  InvalidLimit(ArtifactLimit)
  FileUnavailable(ArtifactFile)
  UnsupportedPathType(ArtifactFile)
  FileTooLarge(file~ : ArtifactFile, maximum~ : Int64)
  TotalTooLarge(maximum~ : Int64)
  FileTruncated(ArtifactFile)
  FileChangedDuringRead(ArtifactFile)
  DigestMismatch(ArtifactFile)
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
/// Explicit paths for the three v1 bundle files.
///
/// LunaFlux does not scan the containing directory or infer filenames.
pub struct ArtifactPaths {
  model_config : String
  tokenizer : String
  weights : String
} derive(Debug, Eq)

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
