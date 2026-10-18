///|
test "paged history separates aggregate storage from per-sequence length" {
  let domain = @attention_physical_ir.PagedHistoryDomain::new(
    rows_per_page=4,
    physical_pages=3,
    maximum_context=20,
    maximum_sequences=8,
  ).unwrap()
  assert_eq(domain.physical_rows(), 12)
  assert_eq(domain.table_capacity(), 40)
  assert_eq(domain.pages_for(0), Some(0))
  assert_eq(domain.pages_for(4), Some(1))
  assert_eq(domain.pages_for(5), Some(2))
  assert_eq(domain.pages_for(20), Some(5))
  assert_eq(domain.pages_for(21), None)
  assert_eq(domain.physical_row(2, 5), Some(9))
  assert_eq(domain.physical_row(0, 19), Some(3))
  assert_eq(domain.physical_row(3, 5), None)
  assert_eq(domain.physical_row(0, 20), None)
}

///|
test "paged history proves index and table extents before multiplication" {
  for
    (rows, pages, context, sequences) in [
      (0, 1, 1, 1),
      (1, 0, 1, 1),
      (1, 1, 0, 1),
      (1, 1, 1, 0),
      (2147483647, 2, 1, 1),
      (1, 1, 2147483647, 2),
    ] {
    assert_true(
      @attention_physical_ir.PagedHistoryDomain::new(
        rows_per_page=rows,
        physical_pages=pages,
        maximum_context=context,
        maximum_sequences=sequences,
      )
      is None,
    )
  }
}
