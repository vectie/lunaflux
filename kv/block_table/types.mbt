///|
const ABSOLUTE_MAX_TABLE_CAPACITY : Int = 1_048_576

///|
const ABSOLUTE_MAX_PAGES_PER_TABLE : Int = 1_048_576

///|
const ABSOLUTE_MAX_TABLE_PAGE_CELLS : Int = 16_777_216

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
}
