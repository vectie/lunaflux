///|
/// Immutable future page-table epoch. A lookup owner retains one page identity;
/// every tile in the epoch borrows only its subrange. Alignment and ownership
/// determine the span, rather than a model-specific refresh interval.
pub struct PagedLookupLookahead {
  priv tile_rows : Int32
  priv page_rows : Int32
  priv tiles : Int32
} derive(Debug, Eq)

///|
pub extend PagedLookupLookahead with Eq::{equal, not_equal}

///|
pub extend PagedLookupLookahead with @debug.Debug::{to_repr}

///|
pub fn PagedLookupLookahead::new(
  tile_rows~ : Int32,
  page_rows~ : Int32,
  start_alignment~ : Int32,
  owners~ : Int32,
) -> PagedLookupLookahead? {
  guard PagedTileEpochs::new(tile_rows~, page_rows~, start_alignment~, owners~)
    is Some(epoch) else {
    return None
  }
  let tiles = owners / epoch.page_count()
  guard tiles > 0 && tile_rows <= 2147483647 / tiles else { return None }
  Some({ tile_rows, page_rows, tiles, })
}

///|
pub fn PagedLookupLookahead::tiles_per_epoch(
  self : PagedLookupLookahead,
) -> Int32 {
  self.tiles
}

///|
pub fn PagedLookupLookahead::rows_per_epoch(
  self : PagedLookupLookahead,
) -> Int32 {
  self.tiles * self.tile_rows
}

///|
pub fn PagedLookupLookahead::page_owner(
  self : PagedLookupLookahead,
  tile : Int32,
  page : Int32,
) -> Int32? {
  let pages = self.tile_rows / self.page_rows
  if tile < 0 || tile >= self.tiles || page < 0 || page >= pages {
    None
  } else {
    Some(tile * pages + page)
  }
}
