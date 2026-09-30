///|
test "scratch reuse is determined by lifetime overlap, not a device threshold" {
  let sequential = SequentialScratch::new(
    producer_bytes=36864L,
    consumer_bytes=32768L,
    overlapping=false,
  ).unwrap()
  assert_true(sequential.reuses_storage())
  assert_eq(sequential.allocation_bytes(), 36864L)
  let overlapping = SequentialScratch::new(
    producer_bytes=36864L,
    consumer_bytes=32768L,
    overlapping=true,
  ).unwrap()
  assert_false(overlapping.reuses_storage())
  assert_eq(overlapping.allocation_bytes(), 69632L)
  assert_eq(
    SequentialScratch::new(
      producer_bytes=9223372036854775807L,
      consumer_bytes=4L,
      overlapping=false,
    ),
    None,
  )
}
