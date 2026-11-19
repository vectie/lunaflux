///|
test "bounded parser retains depth, strings and duplicate-key semantics" {
  let boundary = "[".repeat(32) + "0" + "]".repeat(32)
  ignore(@json_guard.parse_bounded(boundary, max_depth=32))
  try @json_guard.parse_bounded("[" + boundary + "]", max_depth=32) catch {
    @json_guard.DepthLimitExceeded(limit=32) => ()
    _ => fail("unexpected depth failure")
  } noraise {
    _ => fail("expected depth limit")
  }
  assert_eq(
    @json_guard.parse_bounded("{\"a\":1,\"a\":2}", max_depth=1),
    @json.parse("{\"a\":1,\"a\":2}"),
  )
  ignore(@json_guard.parse_bounded("[\"\\\"[{}]\\\\\"]", max_depth=1))
}

///|
test "bounded parser retains syntax validation and rejects invalid limits" {
  try @json_guard.parse_bounded("[}", max_depth=2) catch {
    @json.InvalidChar(_, '}') => ()
    _ => fail("unexpected syntax failure")
  } noraise {
    _ => fail("expected syntax error")
  }
  for limit in [0, -1, 1025] {
    try @json_guard.parse_bounded("{}", max_depth=limit) catch {
      @json_guard.InvalidDepthLimit => ()
      _ => fail("unexpected limit failure")
    } noraise {
      _ => fail("expected invalid limit")
    }
  }
}
