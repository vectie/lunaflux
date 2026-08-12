///|
/// Startup bounds accepted by the fixed-page metadata allocator.
pub(all) enum PageAllocatorLimitRule {
  PageCapacity
  RunCapacity
  ActiveReferences
  CachedReferences
  Generation
} derive(Debug, Eq)

///|
/// The independently accounted owner whose reference operation failed.
pub(all) enum PageReferenceKind {
  Active
  Cached
} derive(Debug, Eq)

///|
/// Internal property checked only by the explicit diagnostic API.
pub(all) enum PageAllocatorInvariantRule {
  CountPartition
  QueueEndpoint
  QueueCycle
  QueueMembership
  QueueCount
  FreeReferences
  AllocatedReferences
  RetiredReferences
  GenerationRange
} derive(Debug, Eq)

///|
/// Bounded, payload-safe allocator failures.
///
/// Payloads contain only fixed rules or opaque page identities. They never
/// contain request data, device addresses, paths, or mutable metadata.
pub(all) suberror PageAllocatorError {
  InvalidLimit(PageAllocatorLimitRule)
  InvalidRunLength
  RunBufferTooSmall
  RunIndexOutOfBounds
  Exhausted
  StalePage(PageId)
  PageNotAllocated(PageId)
  ReferenceOverflow(PageId, PageReferenceKind)
  ReferenceUnderflow(PageId, PageReferenceKind)
  InvariantViolation(PageAllocatorInvariantRule)
} derive(Debug, Eq)
