///|
fn slice(
  spec : @spec.DeepSeekV4ModelSpec,
  first : Int,
  end : Int,
) -> @deepseek_v4_decoder.DeepSeekDecoderSlice raise {
  @deepseek_v4_decoder.DeepSeekDecoderSlice::new(
    spec,
    first_layer=first,
    end_layer=end,
    maximum_rows=3,
    maximum_context=256,
    maximum_layer_state_bytes=16777216L,
    maximum_layer_workspace_bytes=16777216L,
    maximum_layer_expert_bytes=8589934592L,
    accuracy=@precision.Accuracy::new(absolute=0.02, relative=0.02),
  )
}

///|
test "contiguous DeepSeek slice preserves layer geometry and compact expert banks" {
  let spec = @spec.DeepSeekV4ModelSpec::from_official_model_id(
    "deepseek-ai/DeepSeek-V4-Flash-0731",
  )
  let first = slice(spec, 0, 2)
  let next = slice(spec, 2, 4)
  let joined = slice(spec, 0, 4)
  defer first.close()
  defer next.close()
  defer joined.close()
  inspect(first.layer_range(), content="(0, 2)")
  inspect(next.layer_range(), content="(2, 4)")
  assert_eq(
    joined.expert_bank_bytes(),
    first.expert_bank_bytes() + next.expert_bank_bytes(),
  )
  assert_eq(joined.source_bytes(), first.source_bytes() + next.source_bytes())
  assert_true(joined.expert_bank_bytes() > 0L)
  assert_true(joined.source_bytes().length() > 0)
  inspect(joined.launch_count(), content="0")
  inspect(joined.append_descriptors().length(), content="0")
  try joined.submit(live_rows=1) catch {
    @device.DeviceError::Busy(OrderedExecutorEnqueue) => ()
    _ => fail("unexpected submit error")
  } noraise {
    _ => fail("unprepared slice must not execute")
  }
  joined.close()
  try joined.poll() catch {
    @device.DeviceError::InvalidArgument(OrderedExecutorPoll) => ()
    _ => fail("unexpected poll error")
  } noraise {
    _ => fail("closed slice must not poll")
  }
}

///|
test "DeepSeek slice rejects empty foreign and prediction-layer ranges" {
  let spec = @spec.DeepSeekV4ModelSpec::from_official_model_id(
    "deepseek-ai/DeepSeek-V4-Flash-0731",
  )
  for
    (first, end) in [
      (-1, 1),
      (0, 0),
      (2, 1),
      (0, spec.layer_count() + 1),
      (spec.layer_count(), spec.layer_count() + 1),
    ] {
    try slice(spec, first, end) catch {
      @precision.PrecisionPlanError::IllegalPlacement => ()
      _ => fail("unexpected range error")
    } noraise {
      _ => fail("invalid decoder range must not become base execution")
    }
  }
}
