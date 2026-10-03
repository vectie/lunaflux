///|
/// Replicated immutable page facts may be reduced independently by each
/// subgroup, rather than communicated through a mutable workgroup cell. The
/// producer must use the same uniform tile range and immutable page identities
/// in every subgroup; this plan proves that every required page has an owner.
pub struct PagedValidityOwnership {
  priv tile_rows : Int
  priv page_rows : Int
  priv pages : Int
} derive(Debug, Eq)

///|
pub extend PagedValidityOwnership with Eq::{equal, not_equal}

///|
pub extend PagedValidityOwnership with @debug.Debug::{to_repr}

///|
pub fn PagedValidityOwnership::new(
  tile_rows~ : Int,
  page_rows~ : Int,
  start_alignment~ : Int,
  owners~ : Int,
) -> PagedValidityOwnership? {
  guard PagedTileEpochs::new(tile_rows~, page_rows~, start_alignment~, owners~)
    is Some(epoch) else {
    return None
  }
  Some({ tile_rows, page_rows, pages: epoch.page_count(), })
}

///|
pub fn PagedValidityOwnership::page_count(self : PagedValidityOwnership) -> Int {
  self.pages
}

///|
/// Inactive owners, including every owner of an empty tail, are vacuously
/// valid. Operand rings must snapshot this reduction when that operand is
/// staged: a later K refill must not replace the older V validity snapshot.
pub fn PagedValidityOwnership::valid_tile(
  self : PagedValidityOwnership,
  active_rows : Int,
  pages : ReadOnlyArray[Bool],
) -> Bool? {
  guard active_rows >= 0 &&
    active_rows <= self.tile_rows &&
    pages.length() == self.pages else {
    return None
  }
  for owner in 0..<self.pages {
    if owner * self.page_rows < active_rows && !pages[owner] {
      return Some(false)
    }
  }
  Some(true)
}
