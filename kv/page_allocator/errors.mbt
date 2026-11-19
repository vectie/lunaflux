///|
/// Startup bounds accepted by the fixed-page metadata allocator.
pub(all) enum PageAllocatorLimitRule {
  PageCapacity
  StorageCapacity
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
/// Monotonic identities used by allocator-owned transactions.
pub(all) enum PageAllocatorIdentity {
  AllocationCheckpointEpochIdentity
  PageGenerationIdentity
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
  IdentityOverflow(PageAllocatorIdentity)
  InvalidRunLength
  RunBufferTooSmall
  RunIndexOutOfBounds
  StorageIndexOutOfBounds
  StorageCellEmpty
  AllocationCheckpointAlreadyOpen
  AllocationCheckpointReferenceMutationRejected
  PageAllocationCheckpointMismatch
  StalePageAllocationCheckpointEpoch
  Exhausted
  StalePage(PageId)
  PageNotAllocated(PageId)
  PageNotActive(PageId)
  ReferenceOverflow(PageId, PageReferenceKind)
  ReferenceUnderflow(PageId, PageReferenceKind)
  InvariantViolation(PageAllocatorInvariantRule)
} derive(Debug, Eq)

///|
pub extend PageAllocatorLimitRule with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend PageAllocatorLimitRule with Eq::{not_equal, equal}

///|
pub extend PageReferenceKind with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend PageReferenceKind with Eq::{not_equal, equal}

///|
pub extend PageAllocatorIdentity with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend PageAllocatorIdentity with Eq::{not_equal, equal}

///|
pub extend PageAllocatorInvariantRule with @moonbitlang/core/debug.Debug::{
  to_repr,
}

///|
pub extend PageAllocatorInvariantRule with Eq::{not_equal, equal}

///|
pub extend PageAllocatorError with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend PageAllocatorError with Eq::{not_equal, equal}
