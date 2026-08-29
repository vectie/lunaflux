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
  ByteLimitExceeded(limit~ : Int)
  InvalidUtf8
  NestingLimitExceeded(limit~ : Int)
  DuplicateField(depth~ : Int, index~ : Int)
  MalformedJson
  NonObjectRoot
  UnknownField(component~ : JsonComponent, index~ : Int)
  MissingField(component~ : JsonComponent, field~ : JsonField)
  InvalidField(component~ : JsonComponent, field~ : JsonField, index~ : Int)
  UnsupportedSemantics(
    component~ : JsonComponent,
    feature~ : UnsupportedFeature
  )
  VocabularyLimitExceeded(limit~ : Int)
  MergeLimitExceeded(limit~ : Int)
  AddedTokenLimitExceeded(limit~ : Int)
  InvalidByteLevelPiece(entry_index~ : Int, character_index~ : Int)
  AddedTokenNotInVocabulary(index~ : Int)
  AddedTokenContentMismatch(index~ : Int)
  MalformedMerge(index~ : Int)
  UnknownMergePiece(index~ : Int, part~ : MergePart)
  Validation(@tokenizer.TokenizerError)
} derive(Debug, Eq)
