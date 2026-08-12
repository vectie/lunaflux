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
  StorageIndexOutOfBounds
  StorageCellEmpty
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
