///|
fn prediction_slice(
  first : Int32,
  end : Int32,
  enabled : Bool,
) -> @deepseek_v4_decoder.DeepSeekDecoderSlice raise {
  @deepseek_v4_decoder.DeepSeekDecoderSlice::new(
    @spec.DeepSeekV4ModelSpec::from_official_model_id(
      "deepseek-ai/DeepSeek-V4-Flash-0731",
    ),
    first_layer=first,
    end_layer=end,
    maximum_rows=3,
    maximum_context=256,
    maximum_layer_state_bytes=16777216L,
    maximum_layer_workspace_bytes=16777216L,
    maximum_layer_expert_bytes=8589934592L,
    accuracy=@precision.Accuracy::new(absolute=0.02, relative=0.02),
    capture_prediction_inputs=enabled,
  )
}

///|
test "DSpark captured target order survives arbitrary contiguous stage partition" {
  let plain = prediction_slice(39, 43, false)
  let complete = prediction_slice(39, 43, true)
  let left = prediction_slice(39, 41, true)
  let right = prediction_slice(41, 43, true)
  let empty = prediction_slice(0, 1, true)
  defer plain.close()
  defer complete.close()
  defer left.close()
  defer right.close()
  defer empty.close()
  assert_eq(plain.prediction_capture_bytes(), 0L)
  assert_eq(empty.prediction_capture_bytes(), 0L)
  assert_eq(
    complete.prediction_capture_layers(),
    ReadOnlyArray::from_array([40, 41, 42]),
  )
  assert_eq(left.prediction_capture_layers(), ReadOnlyArray::from_array([40]))
  assert_eq(
    right.prediction_capture_layers(),
    ReadOnlyArray::from_array([41, 42]),
  )
  assert_eq(complete.prediction_capture_bytes(), 73728L)
  assert_eq(
    complete.prediction_capture_bytes(),
    left.prediction_capture_bytes() + right.prediction_capture_bytes(),
  )
  let source = @utf8.decode(complete.source_bytes())
  assert_true(source.contains("deepseek_prediction_capture_39_segment_2"))
  assert_false(
    @utf8.decode(plain.source_bytes()).contains("prediction_capture"),
  )
  assert_false(
    @utf8.decode(empty.source_bytes()).contains("prediction_capture"),
  )
  try complete.prediction_input() catch {
    @device.DeviceError::InvalidArgument(OrderedExecutorCreate) => ()
    e => raise e
  } noraise {
    _ => fail("unprepared capture is not an executable input")
  }
}
