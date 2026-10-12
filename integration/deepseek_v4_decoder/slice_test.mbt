///|
test "control laws reach both envelopes without changing default source" {
  let spec = @spec.DeepSeekV4ModelSpec::from_official_model_id(
    "deepseek-ai/DeepSeek-V4-Flash-0731",
  )
  let default = slice(spec, 0, 4)
  defer default.close()
  let refined = @deepseek_v4_decoder.DeepSeekDecoderSlice::new(
    spec,
    first_layer=0,
    end_layer=4,
    maximum_rows=3,
    maximum_context=256,
    maximum_layer_state_bytes=16777216L,
    maximum_layer_workspace_bytes=16777216L,
    maximum_layer_expert_bytes=8589934592L,
    accuracy=@precision.Accuracy::new(absolute=0.02, relative=0.02),
    control_square_law=CompensatedFloat32,
    control_dot_law=StripedTreeFloat32(partitions=32),
  )
  defer refined.close()
  let code = @utf8.decode(refined.source_bytes())
  assert_eq(
    code.split("float square_sum = 0.0f, square_sum_correction = 0.0f").count(),
    9,
  )
  assert_eq(code.split("float dot_partials[32] = {}").count(), 9)
  assert_eq(default.expert_bank_bytes(), refined.expert_bank_bytes())
  let explicit_default = @deepseek_v4_decoder.DeepSeekDecoderSlice::new(
    spec,
    first_layer=0,
    end_layer=4,
    maximum_rows=3,
    maximum_context=256,
    maximum_layer_state_bytes=16777216L,
    maximum_layer_workspace_bytes=16777216L,
    maximum_layer_expert_bytes=8589934592L,
    accuracy=@precision.Accuracy::new(absolute=0.02, relative=0.02),
    control_square_law=OrderedFloat32,
    control_dot_law=OrderedFloat32,
  )
  defer explicit_default.close()
  assert_eq(default.source_bytes(), explicit_default.source_bytes())
}

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
test "output reduction law propagates through window and compressed layers" {
  let spec = @spec.DeepSeekV4ModelSpec::from_official_model_id(
    "deepseek-ai/DeepSeek-V4-Flash-0731",
  )
  let ordinary = slice(spec, 0, 4)
  defer ordinary.close()
  for
    law in [
      @precision.DotReductionLaw::OrderedFloat32,
      StripedTreeFloat32(partitions=32),
    ] {
    let selected = @deepseek_v4_decoder.DeepSeekDecoderSlice::new(
      spec,
      first_layer=0,
      end_layer=4,
      maximum_rows=3,
      maximum_context=256,
      maximum_layer_state_bytes=16777216L,
      maximum_layer_workspace_bytes=16777216L,
      maximum_layer_expert_bytes=8589934592L,
      accuracy=@precision.Accuracy::new(absolute=0.02, relative=0.02),
      output_accumulation=law,
    )
    defer selected.close()
    if law == OrderedFloat32 {
      assert_eq(ordinary.source_bytes(), selected.source_bytes())
    } else {
      let text = @utf8.decode(selected.source_bytes())
      assert_eq(text.split("const int partition = threadIdx.x % 32").count(), 5)
    }
    assert_eq(ordinary.expert_bank_bytes(), selected.expert_bank_bytes())
  }
}

///|
test "explicit attention accumulation reaches window and compressed decoder sources" {
  let spec = @spec.DeepSeekV4ModelSpec::from_official_model_id(
    "deepseek-ai/DeepSeek-V4-Flash-0731",
  )
  let original = slice(spec, 0, 4)
  defer original.close()
  let ordered = @deepseek_v4_decoder.DeepSeekDecoderSlice::new(
    spec,
    first_layer=0,
    end_layer=4,
    maximum_rows=3,
    maximum_context=256,
    maximum_layer_state_bytes=16777216L,
    maximum_layer_workspace_bytes=16777216L,
    maximum_layer_expert_bytes=8589934592L,
    accuracy=@precision.Accuracy::new(absolute=0.02, relative=0.02),
    attention_accumulation=OrderedFloat32,
  )
  defer ordered.close()
  let tiled = @deepseek_v4_decoder.DeepSeekDecoderSlice::new(
    spec,
    first_layer=0,
    end_layer=4,
    maximum_rows=3,
    maximum_context=256,
    maximum_layer_state_bytes=16777216L,
    maximum_layer_workspace_bytes=16777216L,
    maximum_layer_expert_bytes=8589934592L,
    accuracy=@precision.Accuracy::new(absolute=0.02, relative=0.02),
    attention_accumulation=TiledFloat32,
  )
  defer tiled.close()
  assert_eq(original.source_bytes(), ordered.source_bytes())
  let text = @utf8.decode(tiled.source_bytes())
  assert_true(text.contains("wm::mma_sync"))
  assert_true(text.contains("deepseek_l0_window_attention"))
  assert_true(text.contains("deepseek_l3_joint_attention_attention"))
  assert_eq(original.expert_bank_bytes(), tiled.expert_bank_bytes())
}

///|
test "rotary product law reaches main query cache and output without changing default" {
  let spec = @spec.DeepSeekV4ModelSpec::from_official_model_id(
    "deepseek-ai/DeepSeek-V4-Flash-0731",
  )
  let omitted = slice(spec, 0, 4)
  defer omitted.close()
  for
    product in [
      @precision.RotaryProductLaw::SeparateFloat32Products,
      ComplexFloat32Fma,
    ] {
    let selected = @deepseek_v4_decoder.DeepSeekDecoderSlice::new(
      spec,
      first_layer=0,
      end_layer=4,
      maximum_rows=3,
      maximum_context=256,
      maximum_layer_state_bytes=16777216L,
      maximum_layer_workspace_bytes=16777216L,
      maximum_layer_expert_bytes=8589934592L,
      accuracy=@precision.Accuracy::new(absolute=0.02, relative=0.02),
      rotary_product=product,
    )
    defer selected.close()
    if product == SeparateFloat32Products {
      assert_eq(omitted.source_bytes(), selected.source_bytes())
    } else {
      let text = @utf8.decode(selected.source_bytes())
      // Two rotations per window layer; query, pooled KV and output per
      // compressed layer. Indexer/Hadamard arithmetic is a separate policy.
      assert_eq(
        text.split("__fmaf_rn(left, cosine, -(__fmul_rn(right, sine)))").count(),
        11,
      )
      assert_eq(
        text.split("__fmaf_rn(right, cosine, __fmul_rn(left, sine))").count(),
        11,
      )
    }
    assert_eq(omitted.expert_bank_bytes(), selected.expert_bank_bytes())
  }
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
  try
    joined.visit_retired_boundaries((_, _, _) => {
      fail("unexpected borrowed port")
    })
  catch {
    @device.DeviceError::InvalidArgument(OrderedExecutorPoll) => ()
    _ => fail("unexpected observer error")
  } noraise {
    _ => fail("unprepared slice cannot expose diagnostic boundaries")
  }
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
