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
  token_id : Int
  piece : Bytes
  kind : TokenKind
} derive(Debug, Eq)

///|
/// One ranked BPE merge. Array order supplied to `TokenizerSpec::new` is the
/// rank: a smaller index has higher priority.
pub(all) struct MergeRule {
  left_token_id : Int
  right_token_id : Int
  result_token_id : Int
} derive(Debug, Eq)

///|
/// Resource limits enforced while validating and using one tokenizer.
pub(all) struct TokenizerLimits {
  max_input_bytes : Int
  max_output_tokens : Int
  max_decoded_bytes : Int
  max_vocab_entries : Int
  max_merge_rules : Int
  max_token_bytes : Int
  max_special_tokens : Int
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
  token_ids : Array[Int]
  input_bytes : Int
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
  VocabularyTooLarge(actual~ : Int, maximum~ : Int)
  MergeTableTooLarge(actual~ : Int, maximum~ : Int)
  InvalidTokenId(entry_index~ : Int)
  DuplicateTokenId(token_id~ : Int)
  EmptyTokenPiece(entry_index~ : Int)
  TokenPieceTooLong(entry_index~ : Int, actual~ : Int, maximum~ : Int)
  DuplicateTokenPiece(first_index~ : Int, duplicate_index~ : Int)
  TooManySpecialTokens(actual~ : Int, maximum~ : Int)
  MissingByteToken(byte_value~ : Int)
  InvalidMergeToken(rule_index~ : Int)
  SpecialTokenInMerge(rule_index~ : Int)
  InvalidMergeResult(rule_index~ : Int)
  DuplicateMergePair(first_rule~ : Int, duplicate_rule~ : Int)
  InvalidOutputLimit(requested~ : Int, maximum~ : Int)
  InputTooLong(actual~ : Int, maximum~ : Int)
  OutputTooLong(actual~ : Int, maximum~ : Int)
  SpecialTokenRejected(token_id~ : Int)
  UnknownTokenId(token_id~ : Int)
  DecodedOutputTooLong(maximum~ : Int)
  InvalidDecodeDestination
  DecodeDestinationTooSmall(required~ : Int, available~ : Int)
  InvalidUtf8Output
  InvalidUtf8Input(byte_offset~ : Int)
  UnsupportedQwen3Unicode(byte_offset~ : Int)
  InvalidUtf16Input(code_unit_offset~ : Int)
  InvalidSentencePieceToken(entry_index~ : Int)
  InvalidSentencePieceUnknownToken(token_id~ : Int)
  InvalidSentencePieceBosToken(token_id~ : Int)
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
pub struct DecodePieceStatus(Int, Int)

///|
pub fn DecodePieceStatus::is_ready(self : DecodePieceStatus) -> Bool {
  self.0 == 0
}

///|
pub fn DecodePieceStatus::copied_bytes(self : DecodePieceStatus) -> Int {
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
