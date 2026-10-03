///|
/// Checked immutable page origins. The cache extent proves whether an origin
/// may use a narrow representation; representation selection never relaxes
/// the page-capacity bound. Row/component offsets remain affine consumers.
pub struct PagedCopyOrigin {
  priv stride : Int64
  priv capacity : Int64
  priv offset : Int64
  priv narrow : Bool
} derive(Debug, Eq)

///|
pub extend PagedCopyOrigin with Eq::{equal, not_equal}

///|
pub extend PagedCopyOrigin with @debug.Debug::{to_repr}

///|
pub fn PagedCopyOrigin::new(
  page_stride~ : Int64,
  page_capacity~ : Int64,
  within_page_offset? : Int64 = 0L,
) -> PagedCopyOrigin? {
  if page_stride <= 0L ||
    page_capacity <= 0L ||
    within_page_offset < 0L ||
    page_capacity > 9223372036854775807L / page_stride {
    None
  } else {
    let last_base = (page_capacity - 1L) * page_stride
    guard within_page_offset <= 9223372036854775807L - last_base else {
      return None
    }
    Some({
      stride: page_stride,
      capacity: page_capacity,
      offset: within_page_offset,
      narrow: page_capacity * page_stride <= 4294967295L &&
      last_base + within_page_offset <= 4294967295L,
    })
  }
}

///|
pub fn PagedCopyOrigin::is_narrow(self : PagedCopyOrigin) -> Bool {
  self.narrow
}

///|
pub fn PagedCopyOrigin::page_origin(
  self : PagedCopyOrigin,
  page : Int64,
) -> Int64? {
  if page < 0L || page >= self.capacity {
    None
  } else {
    Some(page * self.stride + self.offset)
  }
}
