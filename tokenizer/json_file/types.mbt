///|
/// Bounded tokenizer artifact snapshot and JSON scan limits.
pub struct TokenizerFileLimits {
  priv max_file_bytes : Int
  priv max_json_depth : Int
}

///|
/// Root-free evidence for one exact tokenizer artifact and parser envelope.
pub struct TokenizerFileAdmission {
  priv root_label : String
  priv locator : String
  priv digest : @tokenizer.TokenizerDigest
  priv file_limits : TokenizerFileLimits
  priv tokenizer_limits : @tokenizer.TokenizerLimits
  priv spec : @tokenizer.TokenizerSpec
}

///|
/// Bounded approved tokenizer-file limit categories.
pub(all) enum TokenizerFileLimit {
  FileBytes
  JsonDepth
} derive(Debug, Eq)

///|
/// Payload-safe approved-file operation categories.
pub(all) enum TokenizerFileOperation {
  BindRoot
  ValidateLocator
  Open
  Snapshot
  Close
} derive(Debug, Eq)

///|
/// Payload-safe approved-tokenizer file failures.
pub(all) suberror TokenizerFileError {
  InvalidLimit(TokenizerFileLimit)
  FileUnavailable
  UnsupportedFileType
  FileTooLarge(maximum~ : Int64)
  FileTruncated
  FileChangedDuringRead
  FileBusy(TokenizerFileOperation)
  FileOperationFailed(TokenizerFileOperation)
  DigestMismatch
  InvalidTokenizer
} derive(Debug, Eq)

///|
/// Return or validate one exact admitted policy field.
pub fn TokenizerFileLimits::new(
  max_file_bytes~ : Int,
  max_json_depth~ : Int,
) -> TokenizerFileLimits raise TokenizerFileError {
  if max_file_bytes <= 0 || max_file_bytes > 1_073_741_824 {
    raise InvalidLimit(FileBytes)
  }
  if max_json_depth <= 0 || max_json_depth > 256 {
    raise InvalidLimit(JsonDepth)
  }
  { max_file_bytes, max_json_depth }
}

///|
/// Return or validate one exact admitted policy field.
pub fn TokenizerFileLimits::max_file_bytes(self : TokenizerFileLimits) -> Int {
  self.max_file_bytes
}

///|
/// Return or validate one exact admitted policy field.
pub fn TokenizerFileLimits::max_json_depth(self : TokenizerFileLimits) -> Int {
  self.max_json_depth
}

///|
/// Return the exact independently verified tokenizer digest.
pub fn TokenizerFileAdmission::digest(
  self : TokenizerFileAdmission,
) -> @tokenizer.TokenizerDigest {
  self.digest
}

///|
/// Return the exact immutable tokenizer specification.
pub fn TokenizerFileAdmission::spec(
  self : TokenizerFileAdmission,
) -> @tokenizer.TokenizerSpec {
  self.spec
}

///|
/// Test an exact authority-free source binding without exposing its labels.
pub fn TokenizerFileAdmission::matches_source(
  self : TokenizerFileAdmission,
  root_label : StringView,
  locator : StringView,
  digest : @tokenizer.TokenizerDigest,
  file_limits : TokenizerFileLimits,
  tokenizer_limits : @tokenizer.TokenizerLimits,
) -> Bool {
  tokenizer_file_source_matches(
    self.root_label,
    self.locator,
    self.digest,
    self.file_limits,
    self.tokenizer_limits,
    root_label,
    locator,
    digest,
    file_limits,
    tokenizer_limits,
  )
}

///|
fn tokenizer_file_source_matches(
  admitted_root_label : StringView,
  admitted_locator : StringView,
  admitted_digest : @tokenizer.TokenizerDigest,
  admitted_file_limits : TokenizerFileLimits,
  admitted_tokenizer_limits : @tokenizer.TokenizerLimits,
  expected_root_label : StringView,
  expected_locator : StringView,
  expected_digest : @tokenizer.TokenizerDigest,
  expected_file_limits : TokenizerFileLimits,
  expected_tokenizer_limits : @tokenizer.TokenizerLimits,
) -> Bool {
  admitted_root_label == expected_root_label &&
  admitted_locator == expected_locator &&
  admitted_digest == expected_digest &&
  admitted_file_limits.max_file_bytes == expected_file_limits.max_file_bytes &&
  admitted_file_limits.max_json_depth == expected_file_limits.max_json_depth &&
  admitted_tokenizer_limits == expected_tokenizer_limits
}
