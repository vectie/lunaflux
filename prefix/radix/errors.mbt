///|
pub(all) enum PrefixLimitRule {
  EntryCapacity
  NodeCapacity
  TokenCapacity
  TokensPerEntry
  PageCapacity
  ScopeBytes
  Priority
  TokensPerPage
  ActiveReferences
  Generation
  LayoutVersion
} derive(Debug, Eq)

///|
pub(all) enum PrefixInvariantRule {
  CountPartition
  RootCount
  NodeOwnership
  NodeUsage
  TerminalLink
  PageChain
  CompressedShape
  EntryShape
  GenerationRange
} derive(Debug, Eq)

///|
pub(all) suberror PrefixIndexError {
  InvalidLimit(PrefixLimitRule)
  InvalidTokenCount
  InvalidToken(Int)
  StaleKey
  CachePermissionDenied
  InvalidPageCount
  ResultBufferTooSmall
  ResultIndexOutOfBounds
  Exhausted
  DuplicatePrefix(PrefixEntryId)
  PublishPlanBusy
  StalePublishPlan
  PublishPlanInvalidated
  EvictionPlanBusy
  StaleEvictionPlan
  EvictionPlanInvalidated
  StaleEntry(PrefixEntryId)
  ReferenceOverflow(PrefixEntryId)
  ReferenceUnderflow(PrefixEntryId)
  InvariantViolation(PrefixInvariantRule)
} derive(Debug, Eq)
