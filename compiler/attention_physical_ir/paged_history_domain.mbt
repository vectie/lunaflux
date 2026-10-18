///|
/// Immutable logical-to-physical history domain. Request length bounds the
/// page table, not a private full-context KV allocation per request. All
/// sequences borrow pages from the same physical pool. Allocation, page
/// ownership and retirement remain runtime effects, outside this plan.
pub struct PagedHistoryDomain {
  priv rows_per_page : Int
  priv pages : Int
  priv context : Int
  priv sequences : Int
  priv pages_per_sequence : Int
}

///|
pub fn PagedHistoryDomain::new(
  rows_per_page~ : Int,
  physical_pages~ : Int,
  maximum_context~ : Int,
  maximum_sequences~ : Int,
) -> Self? {
  guard rows_per_page > 0 &&
    physical_pages > 0 &&
    maximum_context > 0 &&
    maximum_sequences > 0 else {
    return None
  }
  let logical_pages = (maximum_context - 1) / rows_per_page + 1
  // Device row indices and host byte-array extents are signed 32-bit. Do
  // arithmetic in Int64 before proving either address representation fits.
  guard rows_per_page.to_int64() * physical_pages.to_int64() <= 2147483647L &&
    logical_pages.to_int64() * maximum_sequences.to_int64() <= 536870911L else {
    return None
  }
  Some({
    rows_per_page,
    pages: physical_pages,
    context: maximum_context,
    sequences: maximum_sequences,
    pages_per_sequence: logical_pages,
  })
}

///|
pub fn PagedHistoryDomain::geometry(self : Self) -> (Int, Int, Int, Int) {
  (self.rows_per_page, self.pages, self.context, self.sequences)
}

///|
pub fn PagedHistoryDomain::table_capacity(self : Self) -> Int {
  self.pages_per_sequence * self.sequences
}

///|
pub fn PagedHistoryDomain::physical_rows(self : Self) -> Int {
  self.pages * self.rows_per_page
}

///|
pub fn PagedHistoryDomain::pages_for(self : Self, tokens : Int) -> Int? {
  guard tokens >= 0 && tokens <= self.context else { return None }
  Some(if tokens == 0 { 0 } else { (tokens - 1) / self.rows_per_page + 1 })
}

///|
/// Pure address law shared by host fixtures and device lowering. `page` is a
/// physical page selected from the sequence's table at position/page_rows.
pub fn PagedHistoryDomain::physical_row(
  self : Self,
  page : Int,
  position : Int,
) -> Int? {
  guard page >= 0 &&
    page < self.pages &&
    position >= 0 &&
    position < self.context else {
    return None
  }
  Some(page * self.rows_per_page + position % self.rows_per_page)
}
