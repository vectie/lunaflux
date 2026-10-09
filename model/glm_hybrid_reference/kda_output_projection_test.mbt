///|
test "KDA output projection uses row-major output-input weights without bias" {
  let limits = GlmHybridReferenceLimits::new(
    maximum_sequence_length=8,
    maximum_head_count=8,
    maximum_head_dimension=8,
    maximum_channel_count=16,
    maximum_vision_patch_count=8,
  )
  let result = kda_output_projection(
    [1.0, 2.0, 3.0],
    [1.0, 0.0, -1.0, 0.5, 1.0, 2.0],
    row_count=1,
    input_width=3,
    output_width=2,
    limits~,
  )
  assert_eq(result.row_count(), 1)
  assert_eq(result.input_width(), 3)
  assert_eq(result.output_width(), 2)
  assert_eq(result.values(), [-2.0, 8.5])
}

///|
test "KDA output projection rejects a truncated row-major weight" {
  let limits = GlmHybridReferenceLimits::new(
    maximum_sequence_length=8,
    maximum_head_count=8,
    maximum_head_dimension=8,
    maximum_channel_count=16,
    maximum_vision_patch_count=8,
  )
  try
    kda_output_projection(
      [1.0, 2.0],
      [1.0],
      row_count=1,
      input_width=2,
      output_width=1,
      limits~,
    )
  catch {
    InvalidShape => ()
    _ => fail("wrong shape error")
  } noraise {
    _ => fail("truncated weights admitted")
  }
}
