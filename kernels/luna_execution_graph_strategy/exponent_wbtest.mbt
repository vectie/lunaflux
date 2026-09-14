///|
test "graph exponent preserves every admitted shape and rejects boundaries" {
  for value in 1..<=1048576 {
    let exponent = graph_bucket_exponent(value)
    let capacity = graph_bucket_value(exponent)
    assert_true(capacity >= value)
    assert_true(exponent == 0 || capacity / 2 < value)
  }
  for value in [-2147483648, -1, 0, 1048577, 2147483647] {
    try graph_bucket_exponent(value) catch {
      InvalidExecutionGraphBucket(Shape) => ()
      _ => fail("unexpected graph exponent error")
    } noraise {
      _ => fail("invalid graph extent was accepted")
    }
  }
}
