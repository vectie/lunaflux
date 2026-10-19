///|
test "full history shares storage independently of context and concurrency" {
  let plan = @attention_physical_ir.ResidentHistoryCapacity::new(
    rows_per_page=16,
    physical_pages=513,
    maximum_context=8192,
    maximum_sequences=16,
    decode_reserve_pages=1,
  ).unwrap()
  assert_eq(plan.domain().physical_rows(), 8208)
  assert_eq(plan.domain().table_capacity(), 8192)
  assert_eq(plan.resident_sequences_at(512), Some(16))
  assert_eq(plan.resident_sequences_at(1024), Some(8))
  assert_eq(plan.resident_sequences_at(2048), Some(4))
  assert_eq(plan.resident_sequences_at(8192), Some(1))
  assert_eq(plan.resident_sequences_at(8193), None)
  assert_eq(plan.resident_sequences_at(0), None)
  assert_eq(plan.decode_reserve_pages(), 1)
}

///|
test "full retained history cannot advertise an unfinishable maximum context" {
  for (pages, reserve) in [(512, 1), (513, 2), (0, 0), (513, -1)] {
    assert_true(
      @attention_physical_ir.ResidentHistoryCapacity::new(
        rows_per_page=16,
        physical_pages=pages,
        maximum_context=8192,
        maximum_sequences=16,
        decode_reserve_pages=reserve,
      )
      is None,
    )
  }
}
