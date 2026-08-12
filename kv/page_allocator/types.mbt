///|
/// Checked startup bounds for one fixed-page metadata allocator.
pub struct PageAllocatorLimits {
  page_capacity : Int
  max_run_pages : Int
  max_active_references_per_page : Int
  max_cached_references_per_page : Int
  max_generation : Int
} derive(Debug, Eq)

///|
/// Opaque generational identity of one physical page slot.
///
/// An identity is valid only while both its index and generation match the
/// allocator. Callers cannot construct or mutate either component. `#valtype`
/// is a production representation invariant: the native backend returns the
/// two scalar components in registers and stores them inline in run buffers.
#valtype
pub struct PageId(Int, Int) derive(Debug, Eq, Hash)

///|
/// Caller-owned, reusable output storage for exact run allocation.
///
/// Construction allocates its fixed backing storage. `allocate_run` mutates
/// that storage and its logical length without returning a mutable alias.
pub struct PageRunBuffer {
  priv pages : FixedArray[PageId]
  priv mut length : Int
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
  active_references : Int
  cached_references : Int
} derive(Debug, Eq)

///|
/// Allocating diagnostic snapshot of the complete metadata owner.
pub struct PageAllocatorSnapshot {
  free_pages : Int
  allocated_pages : Int
  retired_pages : Int
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
  priv page_capacity : Int
  priv max_run_pages : Int
  priv max_active_references_per_page : Int
  priv max_cached_references_per_page : Int
  priv max_generation : Int
  priv generations : FixedArray[Int]
  priv active_references : FixedArray[Int]
  priv cached_references : FixedArray[Int]
  priv residency : FixedArray[Int]
  priv free_next : FixedArray[Int]
  priv mut free_head : Int
  priv mut free_tail : Int
  priv mut free_pages : Int
  priv mut allocated_pages : Int
  priv mut retired_pages : Int
  priv allocation_journal : FixedArray[Int]
  priv allocation_journal_generations : FixedArray[Int]
  priv allocation_journal_next : FixedArray[Int]
  priv mut allocation_journal_length : Int
  priv mut allocation_checkpoint_open : Bool
  priv mut allocation_checkpoint_epoch : UInt64
  priv mut checkpoint_free_head : Int
  priv mut checkpoint_free_tail : Int
  priv mut checkpoint_free_pages : Int
  priv mut checkpoint_allocated_pages : Int
  priv mut checkpoint_retired_pages : Int
}

///|
fn invalid_page_id() -> PageId {
  PageId(-1, 0)
}
