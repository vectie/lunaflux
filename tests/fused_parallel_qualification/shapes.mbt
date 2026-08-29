///|
fn fused_qualification_shape(
  ordinal : Int,
  name : String,
  positions : Array[Int],
  page_table : Array[Int],
) -> FusedQualificationShape raise FusedQualificationError {
  if ordinal < 0 ||
    name.length() == 0 ||
    positions.length() == 0 ||
    positions.length() > 8 ||
    page_table.length() == 0 ||
    page_table.length() > FUSED_QUALIFICATION_TOTAL_PAGES {
    raise InvalidDeclaredShape
  }
  let mut previous = -1
  for position in positions {
    if position < 0 || position >= 32 || position <= previous {
      raise InvalidDeclaredShape
    }
    let logical_page = position / FUSED_QUALIFICATION_TOKENS_PER_PAGE
    if logical_page >= page_table.length() {
      raise InvalidDeclaredShape
    }
    previous = position
  }
  for page in page_table {
    if page < 0 || page >= FUSED_QUALIFICATION_TOTAL_PAGES {
      raise InvalidDeclaredShape
    }
  }
  {
    ordinal,
    name,
    token_count: positions.length(),
    positions: ReadOnlyArray::from_array(positions),
    page_table: ReadOnlyArray::from_array(page_table),
  }
}

///|
/// Returns the fixed boundary matrix used by both scalar and future physical
/// qualification. The set covers one token, a page tail, a cross-page pair,
/// and the admitted eight-token envelope.
pub fn declared_qualification_shapes() -> ReadOnlyArray[FusedQualificationShape] {
  [
    try! fused_qualification_shape(0, "single-origin", [0], [1]),
    try! fused_qualification_shape(1, "single-page-tail", [15], [2]),
    try! fused_qualification_shape(2, "cross-page-pair", [15, 16], [1, 3]),
    try! fused_qualification_shape(
      3,
      "eight-token-page-boundaries",
      [0, 1, 14, 15, 16, 17, 30, 31],
      [0, 2],
    ),
  ]
}

///|
pub fn FusedQualificationShape::step_counts(
  self : FusedQualificationShape,
) -> ReadOnlyArray[Int] {
  [1, 0, 1, self.token_count, self.page_table.length()]
}

///|
pub fn FusedQualificationShape::query_row_offsets(
  self : FusedQualificationShape,
) -> ReadOnlyArray[Int] {
  [0, self.token_count]
}

///|
pub fn FusedQualificationShape::sequence_lengths(
  self : FusedQualificationShape,
) -> ReadOnlyArray[Int] {
  [self.positions[self.positions.length() - 1] + 1]
}

///|
pub fn FusedQualificationShape::page_table_offsets(
  self : FusedQualificationShape,
) -> ReadOnlyArray[Int] {
  [0, self.page_table.length()]
}

///|
pub fn FusedQualificationShape::expected_dispatch_canary(
  self : FusedQualificationShape,
  per_token_canary : UInt,
) -> UInt {
  per_token_canary * self.token_count.reinterpret_as_uint()
}
