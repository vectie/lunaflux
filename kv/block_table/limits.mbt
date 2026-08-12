///|
pub fn BlockTableLimits::new(
  table_capacity~ : Int,
  max_pages_per_table~ : Int,
  tokens_per_page~ : Int,
  max_generation? : Int = ABSOLUTE_MAX_TABLE_GENERATION,
) -> BlockTableLimits raise BlockTableError {
  if table_capacity <= 0 || table_capacity > ABSOLUTE_MAX_TABLE_CAPACITY {
    raise InvalidLimit(TableCapacity)
  }
  if max_pages_per_table <= 0 ||
    max_pages_per_table > ABSOLUTE_MAX_PAGES_PER_TABLE {
    raise InvalidLimit(PagesPerTable)
  }
  if table_capacity > ABSOLUTE_MAX_TABLE_PAGE_CELLS / max_pages_per_table {
    raise InvalidLimit(PageCellCapacity)
  }
  if tokens_per_page <= 0 || tokens_per_page > ABSOLUTE_MAX_TOKENS_PER_PAGE {
    raise InvalidLimit(TokensPerPage)
  }
  if max_generation <= 0 || max_generation > ABSOLUTE_MAX_TABLE_GENERATION {
    raise InvalidLimit(Generation)
  }
  { table_capacity, max_pages_per_table, tokens_per_page, max_generation }
}

///|
pub fn BlockTableLimits::table_capacity(self : BlockTableLimits) -> Int {
  self.table_capacity
}

///|
pub fn BlockTableLimits::max_pages_per_table(self : BlockTableLimits) -> Int {
  self.max_pages_per_table
}

///|
pub fn BlockTableLimits::tokens_per_page(self : BlockTableLimits) -> Int {
  self.tokens_per_page
}

///|
pub fn BlockTableLimits::max_generation(self : BlockTableLimits) -> Int {
  self.max_generation
}
