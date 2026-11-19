///|
/// Whether a vocabulary entry participates in ordinary BPE or is recognized
/// only through an explicit special-token policy.
pub(all) enum TokenKind {
  Normal
  /// One exact raw byte emitted only when the SentencePiece BPE model cannot
  /// represent an input Unicode scalar directly.
  ByteFallback
  /// An exact added-token match which remains visible under decode `Skip`.
  Added
  Special
} derive(Debug, Eq)

///|
/// One immutable input entry used to construct a tokenizer specification.
///
/// `piece` is the BPE model piece for `Normal`, the one decoded byte for
/// `ByteFallback`, and the literal match bytes for `Special`.
pub(all) struct VocabularyEntry {
  token_id : Int32
  piece : Bytes
  kind : TokenKind
} derive(Debug, Eq)

///|
/// One ranked BPE merge. Array order supplied to `TokenizerSpec::new` is the
/// rank: a smaller index has higher priority.
pub(all) struct MergeRule {
  left_token_id : Int32
  right_token_id : Int32
  result_token_id : Int32
} derive(Debug, Eq)

///|
/// Resource limits enforced while validating and using one tokenizer.
pub(all) struct TokenizerLimits {
  max_input_bytes : Int32
  max_output_tokens : Int32
  max_decoded_bytes : Int32
  max_vocab_entries : Int32
  max_merge_rules : Int32
  max_token_bytes : Int32
  max_special_tokens : Int32
} derive(Debug, Eq)

///|
/// Policy for literal special-token byte sequences encountered during encode.
pub(all) enum SpecialTokenPolicy {
  /// Recognize and emit configured special tokens.
  Allow
  /// Reject the first configured special token found in the input.
  Reject
  /// Treat all input bytes as ordinary bytes; no special matching is done.
  AsOrdinary
} derive(Debug, Eq)

///|
/// Behavior when encoded output exceeds the requested token limit.
pub(all) enum TruncationPolicy {
  RejectOverflow
  Truncate
} derive(Debug, Eq)

///|
/// Whether decode includes configured special-token byte sequences.
pub(all) enum DecodeSpecialPolicy {
  Preserve
  Skip
} derive(Debug, Eq)

///|
/// Result of one bounded encode operation.
pub(all) struct EncodeResult {
  token_ids : Array[Int32]
  input_bytes : Int32
  truncated : Bool
} derive(Debug, Eq)

///|
/// Classifies a tokenizer limit without carrying configuration or payload
/// strings into public errors.
pub(all) enum LimitField {
  InputBytes
  OutputTokens
  DecodedBytes
  VocabularyEntries
  MergeRules
  TokenBytes
  SpecialTokens
} derive(Debug, Eq)

///|
/// Typed, payload-safe failures from tokenizer construction and use.
pub(all) suberror TokenizerError {
  InvalidDigest
  InvalidLimit(field~ : LimitField)
  VocabularyTooLarge(actual~ : Int32, maximum~ : Int32)
  MergeTableTooLarge(actual~ : Int32, maximum~ : Int32)
  InvalidTokenId(entry_index~ : Int32)
  DuplicateTokenId(token_id~ : Int32)
  EmptyTokenPiece(entry_index~ : Int32)
  TokenPieceTooLong(entry_index~ : Int32, actual~ : Int32, maximum~ : Int32)
  DuplicateTokenPiece(first_index~ : Int32, duplicate_index~ : Int32)
  TooManySpecialTokens(actual~ : Int32, maximum~ : Int32)
  MissingByteToken(byte_value~ : Int32)
  InvalidMergeToken(rule_index~ : Int32)
  SpecialTokenInMerge(rule_index~ : Int32)
  InvalidMergeResult(rule_index~ : Int32)
  DuplicateMergePair(first_rule~ : Int32, duplicate_rule~ : Int32)
  InvalidOutputLimit(requested~ : Int32, maximum~ : Int32)
  InputTooLong(actual~ : Int32, maximum~ : Int32)
  OutputTooLong(actual~ : Int32, maximum~ : Int32)
  SpecialTokenRejected(token_id~ : Int32)
  UnknownTokenId(token_id~ : Int32)
  DecodedOutputTooLong(maximum~ : Int32)
  InvalidDecodeDestination
  DecodeDestinationTooSmall(required~ : Int32, available~ : Int32)
  InvalidUtf8Output
  InvalidUtf8Input(byte_offset~ : Int32)
  UnsupportedQwen3Unicode(byte_offset~ : Int32)
  InvalidUtf16Input(code_unit_offset~ : Int32)
  InvalidSentencePieceToken(entry_index~ : Int32)
  InvalidSentencePieceUnknownToken(token_id~ : Int32)
  InvalidSentencePieceBosToken(token_id~ : Int32)
  LunaInvalidStepBudget
  LunaInvalidWorkerCapacity
  LunaTokenizerWorkerBusy
  LunaTokenizerWorkerNotReady
  LunaTokenizerWorkerStale
  LunaInvalidTokenCopy
  LunaInvalidInputWrite
  LunaTokenizerWorkerEpochExhausted
} derive(Debug, Eq)

///|
/// Allocation-free single-piece copy result for owner-resident decode paths.
#valtype
pub struct DecodePieceStatus(Int32, Int32)

///|
pub fn DecodePieceStatus::is_ready(self : DecodePieceStatus) -> Bool {
  self.0 == 0
}

///|
pub fn DecodePieceStatus::copied_bytes(self : DecodePieceStatus) -> Int32 {
  if self.0 != 0 {
    return 0
  }
  self.1
}

///|
/// Opaque, validated lowercase SHA-256 identity of the tokenizer artifact.
pub struct TokenizerDigest {
  priv value : String
} derive(Debug, Eq)

///|
/// Validate a lowercase 64-character SHA-256 hexadecimal identity.
pub fn TokenizerDigest::parse(
  value : String,
) -> TokenizerDigest raise TokenizerError {
  if value.length() != 64 {
    raise InvalidDigest
  }
  for ch in value {
    if !((ch >= '0' && ch <= '9') || (ch >= 'a' && ch <= 'f')) {
      raise InvalidDigest
    }
  }
  { value, }
}

///|
/// Return the validated digest text.
pub fn TokenizerDigest::as_string(self : TokenizerDigest) -> String {
  self.value
}

///|
pub extend TokenKind with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend TokenKind with Eq::{not_equal, equal}

///|
pub extend VocabularyEntry with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend VocabularyEntry with Eq::{not_equal, equal}

///|
pub extend MergeRule with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend MergeRule with Eq::{not_equal, equal}

///|
pub extend TokenizerLimits with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend TokenizerLimits with Eq::{not_equal, equal}

///|
pub extend SpecialTokenPolicy with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend SpecialTokenPolicy with Eq::{not_equal, equal}

///|
pub extend TruncationPolicy with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend TruncationPolicy with Eq::{not_equal, equal}

///|
pub extend DecodeSpecialPolicy with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend DecodeSpecialPolicy with Eq::{not_equal, equal}

///|
pub extend EncodeResult with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend EncodeResult with Eq::{not_equal, equal}

///|
pub extend LimitField with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend LimitField with Eq::{not_equal, equal}

///|
pub extend TokenizerError with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend TokenizerError with Eq::{not_equal, equal}

///|
pub extend TokenizerDigest with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend TokenizerDigest with Eq::{not_equal, equal}
