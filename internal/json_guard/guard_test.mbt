///|
test "accepts distinct nested object keys" {
  @json_guard.reject_duplicate_object_keys(
    "{\"outer\":{\"a\":1},\"items\":[{\"a\":2},{\"b\":3}]}",
    max_depth=4,
  )
}

///|
test "rejects nested duplicate keys" {
  try
    @json_guard.reject_duplicate_object_keys(
      "{\"outer\":{\"a\":1,\"a\":2}}",
      max_depth=4,
    )
  catch {
    error =>
      debug_inspect(error, content="DuplicateObjectKey(depth=2, index=1)")
  } noraise {
    _ => fail("expected nested duplicate rejection")
  }
}

///|
test "rejects escaped-equivalent duplicate keys" {
  try
    @json_guard.reject_duplicate_object_keys(
      "{\"schema_version\":1,\"schema_\\u0076ersion\":2}",
      max_depth=2,
    )
  catch {
    error =>
      debug_inspect(error, content="DuplicateObjectKey(depth=1, index=1)")
  } noraise {
    _ => fail("expected escaped duplicate rejection")
  }
}

///|
test "rejects depth exhaustion and malformed escapes safely" {
  try @json_guard.reject_duplicate_object_keys("[[[]]]", max_depth=2) catch {
    error => debug_inspect(error, content="DepthLimitExceeded(limit=2)")
  } noraise {
    _ => fail("expected depth rejection")
  }
  try
    @json_guard.reject_duplicate_object_keys("{\"\\uD800\":1}", max_depth=2)
  catch {
    error => debug_inspect(error, content="MalformedStructure")
  } noraise {
    _ => fail("expected malformed escape rejection")
  }
}
