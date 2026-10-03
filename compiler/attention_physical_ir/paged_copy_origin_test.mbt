///|
test "retained page origins preserve bounds across narrow and wide layouts" {
  let narrow = @attention_physical_ir.PagedCopyOrigin::new(
    page_stride=8192L,
    page_capacity=16384L,
  ).unwrap()
  assert_true(narrow.is_narrow())
  assert_eq(narrow.page_origin(0L), Some(0L))
  assert_eq(narrow.page_origin(16383L), Some(16383L * 8192L))
  assert_eq(narrow.page_origin(-1L), None)
  assert_eq(narrow.page_origin(16384L), None)
  let wide = @attention_physical_ir.PagedCopyOrigin::new(
    page_stride=8192L,
    page_capacity=524288L,
  ).unwrap()
  assert_false(wide.is_narrow())
  assert_eq(wide.page_origin(524287L), Some(524287L * 8192L))
  for
    (stride, capacity) in [
      (0L, 1L),
      (1L, 0L),
      (-1L, 1L),
      (9223372036854775807L, 2L),
    ] {
    assert_eq(
      @attention_physical_ir.PagedCopyOrigin::new(
        page_stride=stride,
        page_capacity=capacity,
      ),
      None,
    )
  }
  for extent in [4294967295L, 4294967296L] {
    assert_eq(
      @attention_physical_ir.PagedCopyOrigin::new(
        page_stride=extent,
        page_capacity=1L,
      )
      .unwrap()
      .is_narrow(),
      extent == 4294967295L,
    )
  }
  let offset = @attention_physical_ir.PagedCopyOrigin::new(
    page_stride=8192L,
    page_capacity=16384L,
    within_page_offset=896L,
  ).unwrap()
  for page in [0L, 17L, 16383L] {
    for row in [0L, 7L] {
      for component in [0L, 127L] {
        assert_eq(
          offset.page_origin(page).unwrap() + row * 1024L + component,
          page * 8192L + (row * 8L + 7L) * 128L + component,
        )
      }
    }
  }
  assert_false(
    @attention_physical_ir.PagedCopyOrigin::new(
      page_stride=1L,
      page_capacity=1L,
      within_page_offset=4294967296L,
    )
    .unwrap()
    .is_narrow(),
  )
  for within_page_offset in [-1L, 9223372036854775807L] {
    assert_eq(
      @attention_physical_ir.PagedCopyOrigin::new(
        page_stride=8192L,
        page_capacity=2L,
        within_page_offset~,
      ),
      None,
    )
  }
}
