///|
/// Immutable page ownership for an aligned tile. Each page is loaded once
/// per owner group; rows retain their physical page identity and offset.
pub struct PagedTileEpochs {
  priv pages : Int32
  priv page_rows : Int32
} derive(Debug, Eq)

///|
pub extend PagedTileEpochs with Eq::{equal, not_equal}

///|
pub extend PagedTileEpochs with @debug.Debug::{to_repr}

///|
pub fn PagedTileEpochs::new(
  tile_rows~ : Int32,
  page_rows~ : Int32,
  start_alignment~ : Int32,
  owners~ : Int32,
) -> PagedTileEpochs? {
  guard tile_rows > 0 &&
    page_rows > 0 &&
    start_alignment > 0 &&
    owners > 0 &&
    tile_rows % page_rows == 0 &&
    start_alignment % page_rows == 0 &&
    tile_rows / page_rows <= owners else {
    return None
  }
  Some({ pages: tile_rows / page_rows, page_rows, })
}

///|
pub fn PagedTileEpochs::page_count(self : PagedTileEpochs) -> Int32 {
  self.pages
}

///|
pub fn PagedTileEpochs::page_owner(
  self : PagedTileEpochs,
  row : Int32,
) -> Int32? {
  if row < 0 || row / self.page_rows >= self.pages {
    None
  } else {
    Some(row / self.page_rows)
  }
}
