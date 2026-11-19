///|
const ABSOLUTE_MAX_PAGE_CAPACITY : Int32 = 1_048_576

///|
const ABSOLUTE_MAX_PAGE_ID_STORAGE_CAPACITY : Int32 = 16_777_216

///|
const ABSOLUTE_MAX_REFERENCES : Int32 = 1_073_741_823

///|
const ABSOLUTE_MAX_GENERATION : Int32 = 2_147_483_647

///|
pub fn PageAllocatorLimits::new(
  page_capacity~ : Int32,
  max_run_pages~ : Int32,
  max_active_references_per_page~ : Int32,
  max_cached_references_per_page~ : Int32,
  max_generation~ : Int32 = ABSOLUTE_MAX_GENERATION,
) -> PageAllocatorLimits raise PageAllocatorError {
  if page_capacity <= 0 || page_capacity > ABSOLUTE_MAX_PAGE_CAPACITY {
    raise InvalidLimit(PageCapacity)
  }
  if max_run_pages <= 0 || max_run_pages > page_capacity {
    raise InvalidLimit(RunCapacity)
  }
  if max_active_references_per_page <= 0 ||
    max_active_references_per_page > ABSOLUTE_MAX_REFERENCES {
    raise InvalidLimit(ActiveReferences)
  }
  if max_cached_references_per_page <= 0 ||
    max_cached_references_per_page > ABSOLUTE_MAX_REFERENCES {
    raise InvalidLimit(CachedReferences)
  }
  if max_generation <= 0 || max_generation > ABSOLUTE_MAX_GENERATION {
    raise InvalidLimit(Generation)
  }
  {
    page_capacity,
    max_run_pages,
    max_active_references_per_page,
    max_cached_references_per_page,
    max_generation,
  }
}

///|
pub fn PageAllocatorLimits::page_capacity(self : PageAllocatorLimits) -> Int32 {
  self.page_capacity
}

///|
pub fn PageAllocatorLimits::max_run_pages(self : PageAllocatorLimits) -> Int32 {
  self.max_run_pages
}

///|
pub fn PageAllocatorLimits::max_active_references_per_page(
  self : PageAllocatorLimits,
) -> Int32 {
  self.max_active_references_per_page
}

///|
pub fn PageAllocatorLimits::max_cached_references_per_page(
  self : PageAllocatorLimits,
) -> Int32 {
  self.max_cached_references_per_page
}

///|
pub fn PageAllocatorLimits::max_generation(self : PageAllocatorLimits) -> Int32 {
  self.max_generation
}
