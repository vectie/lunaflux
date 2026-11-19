///|
/// Checked startup bounds for one fixed-page metadata allocator.
pub struct PageAllocatorLimits {
  page_capacity : Int32
  max_run_pages : Int32
  max_active_references_per_page : Int32
  max_cached_references_per_page : Int32
  max_generation : Int32
} derive(Debug, Eq)

///|
/// Opaque generational identity of one physical page slot.
///
/// An identity is valid only while both its index and generation match the
/// allocator. Callers cannot construct or mutate either component. `#valtype`
/// is a production representation invariant: the native backend returns the
/// two scalar components in registers and stores them inline in run buffers.
#valtype
pub struct PageId(Int32, Int32) derive(Debug, Eq, Hash)

///|
/// Caller-owned, reusable output storage for exact run allocation.
///
/// Construction allocates its fixed backing storage. `allocate_run` mutates
/// that storage and its logical length without returning a mutable alias.
pub struct PageRunBuffer {
  priv pages : FixedArray[PageId]
  priv mut length : Int32
}

///|
/// Opaque fixed storage for canonical inline page identities.
///
/// The allocator package initializes unused cells with its private invalid
/// sentinel. The public API never returns that sentinel: reads require an
/// occupied cell. Construction allocates both fixed arrays; successful
/// reads, writes, clears, and occupancy checks grow no collection.
pub struct PageIdStorage {
  priv pages : FixedArray[PageId]
  priv occupied : FixedArray[Bool]
}

///|
/// Authenticated, single-use boundary for provisional page allocations.
///
/// The handle is valid only for its owning allocator and current checkpoint
/// epoch. It contains no mutable metadata aliases.
#valtype
pub struct PageAllocationCheckpoint(PageAllocator, UInt64)

///|
/// Diagnostic residency of one physical page slot.
pub(all) enum PageResidency {
  Free
  Allocated
  Retired
} derive(Debug, Eq)

///|
/// Immutable diagnostic copy of one page's metadata.
pub struct PageSnapshot {
  id : PageId
  residency : PageResidency
  active_references : Int32
  cached_references : Int32
} derive(Debug, Eq)

///|
/// Allocating diagnostic snapshot of the complete metadata owner.
pub struct PageAllocatorSnapshot {
  free_pages : Int32
  allocated_pages : Int32
  retired_pages : Int32
  pages : ReadOnlyArray[PageSnapshot]
} derive(Debug, Eq)

///|
/// Single owner of preallocated host metadata for fixed physical pages.
///
/// The arrays and intrusive FIFO are fully allocated by `new`. Successful
/// allocation and reference operations resize no collection and construct no
/// strings, maps, snapshots, or returned arrays.
pub struct PageAllocator {
  priv limits : PageAllocatorLimits
  priv page_capacity : Int32
  priv max_run_pages : Int32
  priv max_active_references_per_page : Int32
  priv max_cached_references_per_page : Int32
  priv max_generation : Int32
  priv generations : FixedArray[Int32]
  priv active_references : FixedArray[Int32]
  priv cached_references : FixedArray[Int32]
  priv residency : FixedArray[Int32]
  priv free_next : FixedArray[Int32]
  priv mut free_head : Int32
  priv mut free_tail : Int32
  priv mut free_pages : Int32
  priv mut allocated_pages : Int32
  priv mut retired_pages : Int32
  priv allocation_journal : FixedArray[Int32]
  priv allocation_journal_generations : FixedArray[Int32]
  priv allocation_journal_next : FixedArray[Int32]
  priv mut allocation_journal_length : Int32
  priv mut allocation_checkpoint_open : Bool
  priv mut allocation_checkpoint_epoch : UInt64
  priv mut checkpoint_free_head : Int32
  priv mut checkpoint_free_tail : Int32
  priv mut checkpoint_free_pages : Int32
  priv mut checkpoint_allocated_pages : Int32
  priv mut checkpoint_retired_pages : Int32
}

///|
fn invalid_page_id() -> PageId {
  PageId(-1, 0)
}

///|
pub extend PageAllocatorLimits with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend PageAllocatorLimits with Eq::{not_equal, equal}

///|
pub extend PageId with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend PageId with Eq::{not_equal, equal}

///|
pub extend PageId with Hash::{hash, hash_combine}

///|
pub extend PageResidency with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend PageResidency with Eq::{not_equal, equal}

///|
pub extend PageSnapshot with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend PageSnapshot with Eq::{not_equal, equal}

///|
pub extend PageAllocatorSnapshot with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend PageAllocatorSnapshot with Eq::{not_equal, equal}
