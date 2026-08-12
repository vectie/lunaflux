///|
test "layout is deterministic, aligned, and indexed by TensorRef" {
  let bindings = test_bindings()
  let layout = @device_materialize.plan_layout(
    bindings,
    alignment_bytes=64L,
    max_arena_bytes=4096L,
  )
  assert_eq(layout.identity(), bindings.identity())
  assert_eq(layout.alignment_bytes(), 64L)
  assert_eq(layout.materialized_bytes(), 392L)
  assert_eq(layout.arena_bytes(), 768L)
  assert_eq(layout.length(), 12)
  let offsets : Array[Int64] = []
  for reference in 0..<layout.length() {
    let tensor_reference = try! @plan.TensorRef::new(reference)
    guard layout.region(tensor_reference) is Some(region) else {
      fail("expected region")
    }
    assert_eq(region.reference(), tensor_reference)
    assert_eq(region.offset_bytes() % layout.alignment_bytes(), 0L)
    assert_true(region.byte_length() > 0L)
    assert_true(region.end_offset_bytes() <= layout.arena_bytes())
    offsets.push(region.offset_bytes())
  }
  assert_eq(offsets, [
    0L, 64L, 128L, 192L, 256L, 320L, 384L, 448L, 512L, 576L, 640L, 704L,
  ])
  assert_eq(
    @device_materialize.plan_layout(
      bindings,
      alignment_bytes=64L,
      max_arena_bytes=4096L,
    ),
    layout,
  )
}

///|
test "layout bounds fail closed before device work" {
  let bindings = test_bindings()
  try
    @device_materialize.plan_layout(
      bindings,
      alignment_bytes=0L,
      max_arena_bytes=4096L,
    )
  catch {
    InvalidAlignment => ()
    _ => fail("unexpected alignment error")
  } noraise {
    _ => fail("expected invalid alignment")
  }
  for alignment in [8L, 24L, 8192L] {
    try
      @device_materialize.plan_layout(
        bindings,
        alignment_bytes=alignment,
        max_arena_bytes=4096L,
      )
    catch {
      InvalidAlignment => ()
      _ => fail("unexpected execution-alignment error")
    } noraise {
      _ => fail("expected execution-alignment rejection")
    }
  }
  try
    @device_materialize.plan_layout(
      bindings,
      alignment_bytes=64L,
      max_arena_bytes=767L,
    )
  catch {
    ArenaTooLarge => ()
    _ => fail("unexpected arena error")
  } noraise {
    _ => fail("expected arena bound failure")
  }
  try
    @device_materialize.plan_layout(
      bindings,
      alignment_bytes=64L,
      max_arena_bytes=0L,
    )
  catch {
    InvalidArenaLimit => ()
    _ => fail("unexpected arena-limit error")
  } noraise {
    _ => fail("expected invalid arena limit")
  }
}
