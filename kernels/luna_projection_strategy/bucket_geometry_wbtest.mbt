///|
test "projection geometric buckets cover every supported row count minimally" {
  for rows in 1..<=1048576 {
    let slot = projection_row_bucket_slot(rows)
    let capacity = projection_bucket_for_slot(slot)
    assert_true(capacity >= rows)
    assert_eq(projection_row_bucket_slot(capacity), slot)
    if slot > 0 {
      assert_true(projection_bucket_for_slot(slot - 1) < rows)
    }
  }
  for invalid in [-1, 0, 1048577, 2147483647] {
    try projection_row_bucket_slot(invalid) catch {
      InvalidProjectionStrategy(Bucket) => ()
      _ => fail("unexpected error")
    } noraise {
      _ => fail("invalid row count accepted")
    }
  }
  for invalid in [-1, 19, 2147483647] {
    try projection_bucket_for_slot(invalid) catch {
      InvalidProjectionStrategy(Bucket) => ()
      _ => fail("unexpected error")
    } noraise {
      _ => fail("invalid slot accepted")
    }
  }
}
