///|
const ABSOLUTE_MAX_TABLE_CAPACITY : Int32 = 1_048_576

///|
const ABSOLUTE_MAX_PAGES_PER_TABLE : Int32 = 1_048_576

///|
const ABSOLUTE_MAX_TABLE_PAGE_CELLS : Int32 = 16_777_216

///|
const ABSOLUTE_MAX_BLOCK_TABLE_ID_STORAGE_CAPACITY : Int32 = 16_777_216

///|
const ABSOLUTE_MAX_TOKENS_PER_PAGE : Int32 = 1_048_576

///|
const ABSOLUTE_MAX_TABLE_GENERATION : Int32 = 2_147_483_647

///|
/// Opaque identity of one request block-table slot.
///
/// Releasing and later reusing a slot changes its generation, so a delayed
/// scheduler or worker message cannot address the new table through an old ID.
#valtype
pub struct BlockTableId(Int32, Int32) derive(Debug, Eq, Hash)

///|
/// Opaque fixed storage for canonical inline block-table identities.
///
/// Unused cells contain a package-private inert sentinel. Reads require an
/// occupied cell, so the sentinel never crosses the public boundary.
/// Construction allocates both fixed arrays; successful reads, writes, clears,
/// and occupancy checks grow no collection.
pub struct BlockTableIdStorage {
  priv tables : FixedArray[BlockTableId]
  priv occupied : FixedArray[Bool]
}

///|
/// Authenticated, single-use boundary for provisional table allocations and
/// reversible mapping suffixes.
#valtype
pub struct BlockTableAllocationCheckpoint(BlockTableArena, UInt64)

///|
/// Checked startup bounds for one dense block-table arena.
pub struct BlockTableLimits {
  table_capacity : Int32
  max_pages_per_table : Int32
  tokens_per_page : Int32
  max_generation : Int32
} derive(Debug, Eq)

///|
/// Reusable caller-owned storage for copied or drained physical page IDs.
///
/// Construction allocates the fixed backing storage. Arena operations update
/// its logical length only after all validation succeeds.
pub struct BlockTablePageBuffer {
  priv pages : @page_allocator.PageIdStorage
  priv mut length : Int32
}

///|
/// Reusable caller-owned output for one exact token-position resolution.
///
/// The canonical page identity remains inline in preallocated storage. A
/// successful resolution updates scalar metadata without returning a boxed
/// aggregate.
pub struct BlockTablePositionBuffer {
  priv page_storage : @page_allocator.PageIdStorage
  priv mut logical_page_index : Int32
  priv mut token_offset : Int32
  priv mut is_set : Bool
}

///|
/// Diagnostic lifecycle state of one table slot.
pub(all) enum BlockTableResidency {
  Free
  Allocated
  Retired
} derive(Debug, Eq)

///|
/// Immutable diagnostic copy of one table slot's bounded metadata.
pub struct BlockTableSlotSnapshot {
  id : BlockTableId
  residency : BlockTableResidency
  page_count : Int32
} derive(Debug, Eq)

///|
/// Allocating diagnostic snapshot. It is excluded from scheduling hot paths.
pub struct BlockTableSnapshot {
  free_tables : Int32
  allocated_tables : Int32
  retired_tables : Int32
  slots : ReadOnlyArray[BlockTableSlotSnapshot]
} derive(Debug, Eq)

///|
/// Single owner of fixed-capacity, per-request logical KV block tables.
///
/// The arena owns mappings only. It neither retains nor releases physical
/// pages and never owns device memory. All metadata and page cells are
/// preallocated by `new`; successful steady-state mutations grow no collection.
pub struct BlockTableArena {
  priv limits : BlockTableLimits
  priv table_capacity : Int32
  priv max_pages_per_table : Int32
  priv tokens_per_page : Int32
  priv max_generation : Int32
  priv states : FixedArray[Int32]
  priv generations : FixedArray[Int32]
  priv page_counts : FixedArray[Int32]
  priv pages : @page_allocator.PageIdStorage
  priv free_next : FixedArray[Int32]
  priv mut free_head : Int32
  priv mut free_tail : Int32
  priv mut free_tables : Int32
  priv mut allocated_tables : Int32
  priv mut retired_tables : Int32
  priv allocation_journal : FixedArray[Int32]
  priv allocation_journal_generations : FixedArray[Int32]
  priv allocation_journal_next : FixedArray[Int32]
  priv provisional_tables : FixedArray[Bool]
  priv mapping_journal_indices : FixedArray[Int32]
  priv mapping_journal_baselines : FixedArray[Int32]
  priv mapping_touched : FixedArray[Bool]
  priv mut allocation_journal_length : Int32
  priv mut mapping_journal_length : Int32
  priv mut allocation_checkpoint_open : Bool
  priv mut allocation_checkpoint_epoch : UInt64
  priv mut checkpoint_free_head : Int32
  priv mut checkpoint_free_tail : Int32
  priv mut checkpoint_free_tables : Int32
  priv mut checkpoint_allocated_tables : Int32
  priv mut checkpoint_retired_tables : Int32
}

///|
fn invalid_block_table_id() -> BlockTableId {
  BlockTableId(-1, 0)
}

///|
pub extend BlockTableId with @debug.Debug::{to_repr}

///|
pub extend BlockTableId with Eq::{not_equal, equal}

///|
pub extend BlockTableId with Hash::{hash, hash_combine}

///|
pub extend BlockTableLimits with @debug.Debug::{to_repr}

///|
pub extend BlockTableLimits with Eq::{not_equal, equal}

///|
pub extend BlockTableResidency with @debug.Debug::{to_repr}

///|
pub extend BlockTableResidency with Eq::{not_equal, equal}

///|
pub extend BlockTableSlotSnapshot with @debug.Debug::{to_repr}

///|
pub extend BlockTableSlotSnapshot with Eq::{not_equal, equal}

///|
pub extend BlockTableSnapshot with @debug.Debug::{to_repr}

///|
pub extend BlockTableSnapshot with Eq::{not_equal, equal}
