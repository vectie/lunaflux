///|
test "liveness planning deterministically reuses aligned activation slots" {
  let source = memory_static_plan()
  let result = @device_memory.plan(source, memory_limits())
  assert_eq(result.identity(), source.identity())
  assert_eq(result.target(), source.target())
  assert_eq(result.token_rows(), 6L)
  assert_eq(result.activation_count(), 5)
  assert_eq(result.operation_count(), 5)
  assert_eq(result.slot_count(), 3)
  assert_eq(result.activation_bytes(), 192L)
  assert_eq(result.workspace().offset_bytes(), 192L)
  assert_eq(result.workspace().byte_length(), 32L)
  assert_eq(result.workspace().alignment_bytes(), 16L)
  assert_eq(result.arena_bytes(), 224L)
  let offsets : Array[Int64] = []
  let slots : Array[Int] = []
  let live_through : Array[Int] = []
  for reference in 0..<result.activation_count() {
    guard result.region(@plan.ActivationRef::new(reference)) is Some(region) else {
      fail("missing activation region")
    }
    assert_eq(region.reference().as_int(), reference)
    assert_eq(region.logical_bytes(), 48L)
    assert_eq(region.reserved_bytes(), 64L)
    assert_eq(region.offset_bytes() % 64L, 0L)
    assert_true(region.end_offset_bytes() <= result.activation_bytes())
    offsets.push(region.offset_bytes())
    slots.push(region.slot().as_int())
    live_through.push(region.live_through_step())
  }
  assert_eq(offsets, [0L, 64L, 128L, 64L, 0L])
  assert_eq(slots, [0, 1, 2, 1, 0])
  assert_eq(live_through, [3, 2, 3, 4, 5])
  assert_eq(@device_memory.plan(source, memory_limits()), result)
}

///|
test "simultaneously live values never overlap and final output survives" {
  let result = @device_memory.plan(memory_static_plan(), memory_limits())
  for left_index in 0..<result.activation_count() {
    let left = result.region(@plan.ActivationRef::new(left_index)).unwrap()
    for right_index in (left_index + 1)..<result.activation_count() {
      let right = result.region(@plan.ActivationRef::new(right_index)).unwrap()
      let lifetimes_overlap = left.produced_by().as_int() <=
        right.live_through_step() &&
        right.produced_by().as_int() <= left.live_through_step()
      if lifetimes_overlap {
        assert_true(
          left.end_offset_bytes() <= right.offset_bytes() ||
          right.end_offset_bytes() <= left.offset_bytes(),
        )
      }
    }
  }
  let final_reference = result.final_output()
  let final_region = result.region(final_reference).unwrap()
  assert_true(final_region.is_retained_output())
  assert_eq(final_region.live_through_step(), result.operation_count())
  assert_eq(final_region.offset_bytes(), 0L)
}

///|
test "operation bindings are immutable semantic slices of the shared arena" {
  let result = @device_memory.plan(memory_static_plan(), memory_limits())
  let residual = result.operation_binding(@plan.OperationId::new(3)).unwrap()
  assert_eq(residual.operation_id().as_int(), 3)
  assert_eq(residual.activation_inputs().length(), 2)
  assert_eq(residual.activation_inputs()[0].reference().as_int(), 0)
  assert_eq(residual.activation_inputs()[1].reference().as_int(), 2)
  assert_eq(residual.activation_outputs().length(), 1)
  assert_eq(residual.activation_outputs()[0].reference().as_int(), 3)
  assert_eq(residual.activation_outputs()[0].offset_bytes(), 64L)
  for index in 0..<result.operation_count() {
    let binding = result
      .operation_binding(@plan.OperationId::new(index))
      .unwrap()
    assert_eq(binding.workspace(), result.workspace())
  }
}

///|
test "workspace padding obeys its stronger alignment and the total ceiling" {
  let static_plan = memory_static_plan(
    workspace_bytes=256L,
    workspace_alignment=256L,
  )
  let result = @device_memory.plan(static_plan, memory_limits())
  assert_eq(result.activation_bytes(), 192L)
  assert_eq(result.workspace().offset_bytes(), 256L)
  assert_eq(result.workspace().alignment_bytes(), 256L)
  assert_eq(result.workspace().byte_length(), 256L)
  assert_eq(result.arena_bytes(), 512L)
  try
    @device_memory.plan(static_plan, memory_limits(max_arena_bytes=511L))
  catch {
    @device_memory.DeviceMemoryError::ArenaLimitExceeded => ()
    _ => fail("unexpected workspace capacity error")
  } noraise {
    _ => fail("expected aligned workspace to exceed the complete arena bound")
  }
}

///|
test "every v1 operation output is sized from the complete shape envelope" {
  let semantic = @llama.build(memory_test_model())
  let static_plan = @device_plan.build(
    semantic,
    memory_test_layout(),
    memory_catalog_for(semantic),
  )
  let result = @device_memory.plan(static_plan, memory_limits())
  assert_eq(result.token_rows(), 16L)
  let seen_qkv = Ref(false)
  let seen_rotary = Ref(false)
  let seen_attention = Ref(false)
  let seen_mlp = Ref(false)
  let seen_head = Ref(false)
  for index, operation in static_plan.operations() {
    let region = result.region(@plan.ActivationRef::new(index)).unwrap()
    let expected = match operation.kind() {
      QkvProjection => {
        seen_qkv.val = true
        256L
      }
      RotaryEmbedding => {
        seen_rotary.val = true
        256L
      }
      CausalAttention => {
        seen_attention.val = true
        128L
      }
      GatedMlp => {
        seen_mlp.val = true
        128L
      }
      LanguageModelHead => {
        seen_head.val = true
        256L
      }
      Embedding | RmsNorm | OutputProjection | ResidualAdd => 128L
    }
    assert_eq(region.logical_bytes(), expected)
  }
  assert_true(seen_qkv.val)
  assert_true(seen_rotary.val)
  assert_true(seen_attention.val)
  assert_true(seen_mlp.val)
  assert_true(seen_head.val)
}

///|
test "limits overflow and the complete arena ceiling fail closed" {
  for alignment in [0L, 8L, 24L, 131072L] {
    try
      @device_memory.DeviceMemoryLimits::new(
        activation_alignment=alignment,
        max_arena_bytes=4096L,
      )
    catch {
      InvalidLimit(ActivationAlignment) => ()
      _ => fail("unexpected alignment error")
    } noraise {
      _ => fail("expected invalid alignment")
    }
  }
  try
    @device_memory.DeviceMemoryLimits::new(
      activation_alignment=64L,
      max_arena_bytes=0L,
    )
  catch {
    InvalidLimit(ArenaBytes) => ()
    _ => fail("unexpected arena-limit error")
  } noraise {
    _ => fail("expected invalid arena limit")
  }
  try
    @device_memory.plan(
      memory_static_plan(),
      memory_limits(max_arena_bytes=223L),
    )
  catch {
    @device_memory.DeviceMemoryError::ArenaLimitExceeded => ()
    _ => fail("unexpected capacity error")
  } noraise {
    _ => fail("expected complete arena to exceed the bound")
  }
  try
    @device_memory.plan(
      memory_static_plan(
        max_batch_rows=2147483647,
        max_sequence_tokens=2147483647,
      ),
      memory_limits(max_arena_bytes=9223372036854775807L),
    )
  catch {
    @device_memory.DeviceMemoryError::SizeOverflow(operation) =>
      assert_eq(operation.as_int(), 0)
    _ => fail("unexpected overflow error")
  } noraise {
    _ => fail("expected checked activation-size overflow")
  }
}
