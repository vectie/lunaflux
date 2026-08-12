///|
const ABSOLUTE_MAX_TABLE_CAPACITY : Int = 1_048_576

///|
const ABSOLUTE_MAX_PAGES_PER_TABLE : Int = 1_048_576

///|
const ABSOLUTE_MAX_TABLE_PAGE_CELLS : Int = 16_777_216

///|
const ABSOLUTE_MAX_BLOCK_TABLE_ID_STORAGE_CAPACITY : Int = 16_777_216

///|
const ABSOLUTE_MAX_TOKENS_PER_PAGE : Int = 1_048_576

///|
const ABSOLUTE_MAX_TABLE_GENERATION : Int = 2_147_483_647

///|
/// Opaque identity of one request block-table slot.
///
/// Releasing and later reusing a slot changes its generation, so a delayed
/// scheduler or worker message cannot address the new table through an old ID.
#valtype
pub struct BlockTableId(Int, Int) derive(Debug, Eq, Hash)

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
  table_capacity : Int
  max_pages_per_table : Int
  tokens_per_page : Int
  max_generation : Int
} derive(Debug, Eq)

///|
/// Reusable caller-owned storage for copied or drained physical page IDs.
///
/// Construction allocates the fixed backing storage. Arena operations update
/// its logical length only after all validation succeeds.
pub struct BlockTablePageBuffer {
  priv pages : @page_allocator.PageIdStorage
  priv mut length : Int
}

///|
/// Reusable caller-owned output for one exact token-position resolution.
///
/// The canonical page identity remains inline in preallocated storage. A
/// successful resolution updates scalar metadata without returning a boxed
/// aggregate.
pub struct BlockTablePositionBuffer {
  priv page_storage : @page_allocator.PageIdStorage
  priv mut logical_page_index : Int
  priv mut token_offset : Int
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
  page_count : Int
} derive(Debug, Eq)

///|
/// Allocating diagnostic snapshot. It is excluded from scheduling hot paths.
pub struct BlockTableSnapshot {
  free_tables : Int
  allocated_tables : Int
  retired_tables : Int
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
  priv table_capacity : Int
  priv max_pages_per_table : Int
  priv tokens_per_page : Int
  priv max_generation : Int
  priv states : FixedArray[Int]
  priv generations : FixedArray[Int]
  priv page_counts : FixedArray[Int]
  priv pages : @page_allocator.PageIdStorage
  priv free_next : FixedArray[Int]
  priv mut free_head : Int
  priv mut free_tail : Int
  priv mut free_tables : Int
  priv mut allocated_tables : Int
  priv mut retired_tables : Int
  priv allocation_journal : FixedArray[Int]
  priv allocation_journal_generations : FixedArray[Int]
  priv allocation_journal_next : FixedArray[Int]
  priv provisional_tables : FixedArray[Bool]
  priv mapping_journal_indices : FixedArray[Int]
  priv mapping_journal_baselines : FixedArray[Int]
  priv mapping_touched : FixedArray[Bool]
  priv mut allocation_journal_length : Int
  priv mut mapping_journal_length : Int
  priv mut allocation_checkpoint_open : Bool
  priv mut allocation_checkpoint_epoch : UInt64
  priv mut checkpoint_free_head : Int
  priv mut checkpoint_free_tail : Int
  priv mut checkpoint_free_tables : Int
  priv mut checkpoint_allocated_tables : Int
  priv mut checkpoint_retired_tables : Int
}

///|
fn invalid_block_table_id() -> BlockTableId {
  BlockTableId(-1, 0)
}
