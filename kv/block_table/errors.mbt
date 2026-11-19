///|
pub(all) enum BlockTableLimitRule {
  TableCapacity
  StorageCapacity
  PagesPerTable
  PageCellCapacity
  TokensPerPage
  Generation
} derive(Debug, Eq)

///|
/// Monotonic identities owned by the block-table arena.
pub(all) enum BlockTableIdentity {
  AllocationCheckpointEpochIdentity
  TableGenerationIdentity
} derive(Debug, Eq)

///|
pub(all) enum BlockTableInvariantRule {
  CountPartition
  QueueEndpoint
  QueueCycle
  QueueMembership
  QueueCount
  LifecycleState
  TableShape
  ClearedCells
  GenerationRange
} derive(Debug, Eq)

///|
/// Bounded, payload-safe block-table failures.
///
/// Payloads contain only fixed rules or opaque table identities. They never
/// contain tokens, request data, device addresses, paths, or vendor details.
pub(all) suberror BlockTableError {
  InvalidLimit(BlockTableLimitRule)
  IdentityOverflow(BlockTableIdentity)
  StorageIndexOutOfBounds
  StorageCellEmpty
  AllocationCheckpointAlreadyOpen
  AllocationCheckpointMutationRejected
  InvalidCheckpointDetach(BlockTableId)
  BlockTableAllocationCheckpointMismatch
  StaleBlockTableAllocationCheckpointEpoch
  Exhausted
  StaleTable(BlockTableId)
  TableNotAllocated(BlockTableId)
  TableNotEmpty(BlockTableId)
  InvalidPageCount
  InvalidTruncateLength(BlockTableId)
  TableFull(BlockTableId)
  PageBufferTooSmall
  PageBufferIndexOutOfBounds
  PositionUnset
  SnapshotIndexOutOfBounds
  LogicalPageIndexOutOfBounds(BlockTableId)
  TokenPositionOutOfBounds(BlockTableId)
  InvariantViolation(BlockTableInvariantRule)
} derive(Debug, Eq)

///|
pub extend BlockTableLimitRule with @debug.Debug::{to_repr}

///|
pub extend BlockTableLimitRule with Eq::{not_equal, equal}

///|
pub extend BlockTableIdentity with @debug.Debug::{to_repr}

///|
pub extend BlockTableIdentity with Eq::{not_equal, equal}

///|
pub extend BlockTableInvariantRule with @debug.Debug::{to_repr}

///|
pub extend BlockTableInvariantRule with Eq::{not_equal, equal}

///|
pub extend BlockTableError with @debug.Debug::{to_repr}

///|
pub extend BlockTableError with Eq::{not_equal, equal}
