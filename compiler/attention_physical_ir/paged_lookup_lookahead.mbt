///|
/// Immutable future page-table epoch. A lookup owner retains one page identity;
/// every tile in the epoch borrows only its subrange. Alignment and ownership
/// determine the span, rather than a model-specific refresh interval.
pub struct PagedLookupLookahead {
  priv tile_rows : Int
  priv page_rows : Int
  priv tiles : Int
} derive(Debug, Eq)

///|
pub extend PagedLookupLookahead with Eq::{equal, not_equal}

///|
pub extend PagedLookupLookahead with @debug.Debug::{to_repr}

///|
pub fn PagedLookupLookahead::new(
  tile_rows~ : Int,
  page_rows~ : Int,
  start_alignment~ : Int,
  owners~ : Int,
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
) -> Int {
  self.tiles
}

///|
pub fn PagedLookupLookahead::rows_per_epoch(self : PagedLookupLookahead) -> Int {
  self.tiles * self.tile_rows
}

///|
pub fn PagedLookupLookahead::page_owner(
  self : PagedLookupLookahead,
  tile : Int,
  page : Int,
) -> Int? {
  let pages = self.tile_rows / self.page_rows
  if tile < 0 || tile >= self.tiles || page < 0 || page >= pages {
    None
  } else {
    Some(tile * pages + page)
  }
}
