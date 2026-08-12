///|
pub(all) enum PrefixLimitRule {
  EntryCapacity
  NodeCapacity
  TokensPerEntry
  PageCapacity
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
  EntryShape
  GenerationRange
} derive(Debug, Eq)

///|
pub(all) suberror PrefixIndexError {
  InvalidLimit(PrefixLimitRule)
  InvalidTokenCount
  InvalidToken(Int)
  InvalidPageCount
  ResultBufferTooSmall
  ResultIndexOutOfBounds
  Exhausted
  DuplicatePrefix(PrefixEntryId)
  StaleEntry(PrefixEntryId)
  ReferenceOverflow(PrefixEntryId)
  ReferenceUnderflow(PrefixEntryId)
  NoEvictableEntry
  InvariantViolation(PrefixInvariantRule)
} derive(Debug, Eq)
