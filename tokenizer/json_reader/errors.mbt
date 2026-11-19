///|
/// tokenizer.json component associated with a bounded admission failure.
pub(all) enum JsonComponent {
  Root
  Model
  AddedToken
  Normalizer
  PreTokenizer
  PostProcessor
  Decoder
} derive(Debug, Eq)

///|
/// Selected tokenizer.json fields, named without retaining input text.
pub(all) enum JsonField {
  Version
  Truncation
  Padding
  AddedTokens
  NormalizerField
  PreTokenizerField
  PostProcessorField
  DecoderField
  ModelField
  TypeField
  Vocab
  Merges
  Dropout
  UnknownToken
  ContinuingSubwordPrefix
  EndOfWordSuffix
  FuseUnknown
  ByteFallback
  IgnoreMerges
  Id
  Content
  SingleWord
  LeftStrip
  RightStrip
  Normalized
  Special
  AddPrefixSpace
  TrimOffsets
  UseRegex
  Pattern
  Components
  Template
  TypeId
  SpecialTokens
  Start
  Stop
  Behavior
  Invert
} derive(Debug, Eq)

///|
/// Behavior that the raw-byte TokenizerSpec deliberately does not model.
pub(all) enum UnsupportedFeature {
  SchemaVersion
  TruncationBehavior
  PaddingBehavior
  NormalizationBehavior
  PreTokenizerKind
  RegexPreTokenization
  PrefixSpaceInsertion
  PostProcessingBehavior
  DecoderKind
  ModelKind
  MergeDropout
  UnknownTokenFallback
  ContinuingSubwords
  WordSuffixes
  UnknownTokenFusion
  ByteFallbackBehavior
  IgnoreMergeBehavior
  OrdinaryAddedToken
  AddedTokenMatching
  Qwen3PreTokenization
  Qwen3AddedTokens
} derive(Debug, Eq)

///|
/// Which side of a malformed merge row could not be resolved.
pub(all) enum MergePart {
  Left
  Right
  Result
} derive(Debug, Eq)

///|
/// Bounded, payload-safe tokenizer.json admission failures.
///
/// No variant retains field names, token pieces, JSON text, or artifact paths.
pub(all) suberror TokenizerJsonError {
  InvalidByteLimit
  InvalidDepthLimit
  ByteLimitExceeded(limit~ : Int32)
  InvalidUtf8
  NestingLimitExceeded(limit~ : Int32)
  DuplicateField(depth~ : Int32, index~ : Int32)
  MalformedJson
  NonObjectRoot
  UnknownField(component~ : JsonComponent, index~ : Int32)
  MissingField(component~ : JsonComponent, field~ : JsonField)
  InvalidField(component~ : JsonComponent, field~ : JsonField, index~ : Int32)
  UnsupportedSemantics(
    component~ : JsonComponent,
    feature~ : UnsupportedFeature
  )
  VocabularyLimitExceeded(limit~ : Int32)
  MergeLimitExceeded(limit~ : Int32)
  AddedTokenLimitExceeded(limit~ : Int32)
  InvalidByteLevelPiece(entry_index~ : Int32, character_index~ : Int32)
  AddedTokenNotInVocabulary(index~ : Int32)
  AddedTokenContentMismatch(index~ : Int32)
  MalformedMerge(index~ : Int32)
  UnknownMergePiece(index~ : Int32, part~ : MergePart)
  Validation(@tokenizer.TokenizerError)
} derive(Debug, Eq)

///|
pub extend JsonComponent with @debug.Debug::{to_repr}

///|
pub extend JsonComponent with Eq::{not_equal, equal}

///|
pub extend JsonField with @debug.Debug::{to_repr}

///|
pub extend JsonField with Eq::{not_equal, equal}

///|
pub extend UnsupportedFeature with @debug.Debug::{to_repr}

///|
pub extend UnsupportedFeature with Eq::{not_equal, equal}

///|
pub extend MergePart with @debug.Debug::{to_repr}

///|
pub extend MergePart with Eq::{not_equal, equal}

///|
pub extend TokenizerJsonError with @debug.Debug::{to_repr}

///|
pub extend TokenizerJsonError with Eq::{not_equal, equal}
