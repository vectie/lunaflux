///|
test "projection postprocess attention replacement keeps the fast projection" {
  let replaced = replace_projection_postprocess_attention_spans(
    ReadOnlyArray::from_array([0, 1, 2, 3, 4, 5]),
    ReadOnlyArray::from_array([1]),
    ReadOnlyArray::from_array([20]),
    ReadOnlyArray::from_array([30]),
  )
  assert_eq(replaced, [0, 1, 20, 30, 5])
}

///|
test "projection postprocess attention replacement preserves layer order" {
  let replaced = replace_projection_postprocess_attention_spans(
    ReadOnlyArray::from_array([0, 1, 2, 3, 4, 5, 6, 7, 8]),
    ReadOnlyArray::from_array([0, 4]),
    ReadOnlyArray::from_array([20, 21]),
    ReadOnlyArray::from_array([30, 31]),
  )
  assert_eq(replaced, [0, 20, 30, 4, 21, 31, 8])
}

///|
test "projection postprocess attention replacement rejects overlap" {
  try
    replace_projection_postprocess_attention_spans(
      ReadOnlyArray::from_array([0, 1, 2, 3, 4]),
      ReadOnlyArray::from_array([0, 1]),
      ReadOnlyArray::from_array([20, 21]),
      ReadOnlyArray::from_array([30, 31]),
    )
  catch {
    InvalidExecutor(ResourceState) => ()
    _ => fail("unexpected replacement error")
  } noraise {
    _ => fail("overlapping spans reached execution order")
  }
}

///|
test "projection postprocess attention replacement rejects cardinality drift" {
  try
    replace_projection_postprocess_attention_spans(
      ReadOnlyArray::from_array([0, 1, 2, 3]),
      ReadOnlyArray::from_array([0]),
      ReadOnlyArray::from_array([]),
      ReadOnlyArray::from_array([30]),
    )
  catch {
    InvalidExecutor(ResourceState) => ()
    _ => fail("unexpected replacement error")
  } noraise {
    _ => fail("mismatched replacement arrays reached execution order")
  }
}
