///|
test "numeric layout sizes BF16 FP8 I8 and F32 metadata exactly" {
  let plans = [
    numeric_layout_plan(1),
    numeric_layout_plan(2),
    numeric_layout_plan(3),
  ]
  let expected_lengths = [[64L, 64L], [64L, 32L, 4L], [64L, 32L, 32L]]
  let expected_offsets = [[0L, 64L], [0L, 64L, 96L], [0L, 64L, 96L]]
  let expected_materialized = [128L, 100L, 128L]
  let expected_arena = [128L, 100L, 128L]
  for mode, plan in plans {
    let layout = @device_materialize.plan_numeric_layout(
      plan,
      alignment_bytes=16L,
      max_arena_bytes=1024L,
    )
    assert_eq(layout.identity(), plan.identity())
    assert_eq(layout.length(), plan.tensor_count())
    assert_eq(layout.materialized_bytes(), expected_materialized[mode])
    assert_eq(layout.arena_bytes(), expected_arena[mode])
    for index in 0..<layout.length() {
      let reference = try! @plan.TensorRef::new(index)
      guard layout.region(reference) is Some(region) else {
        fail("numeric tensor region is missing")
      }
      assert_eq(region.reference(), reference)
      assert_eq(region.offset_bytes(), expected_offsets[mode][index])
      assert_eq(region.byte_length(), expected_lengths[mode][index])
      assert_eq(
        region.end_offset_bytes(),
        region.offset_bytes() + region.byte_length(),
      )
    }
  }
}

///|
test "numeric layout includes scales after parameters in exact TensorRef order" {
  for mode in [2, 3] {
    let plan = numeric_layout_plan(mode)
    let layout = @device_materialize.plan_numeric_layout(
      plan,
      alignment_bytes=16L,
      max_arena_bytes=1024L,
    )
    assert_eq(layout.length(), 3)
    guard plan.tensor(try! @plan.TensorRef::new(1)) is Some(parameter) else {
      fail("scaled parameter is missing")
    }
    guard parameter.scale() is Some(scale_reference) else {
      fail("scaled parameter lost its associated metadata reference")
    }
    assert_eq(scale_reference.tensor().as_int(), 2)
    guard layout.region(scale_reference.tensor()) is Some(scale_region) else {
      fail("scale metadata was omitted from the arena")
    }
    assert_eq(scale_region.reference(), scale_reference.tensor())
    assert_eq(scale_region.offset_bytes(), 96L)
  }
}

///|
test "numeric layout rejects alignment and arena bounds before publication" {
  let plan = numeric_layout_plan(1)
  for alignment in [0L, 8L, 24L, 8192L] {
    try
      @device_materialize.plan_numeric_layout(
        plan,
        alignment_bytes=alignment,
        max_arena_bytes=1024L,
      )
    catch {
      InvalidAlignment => ()
      _ => fail("unexpected numeric alignment error")
    } noraise {
      _ => fail("invalid numeric alignment was accepted")
    }
  }
  try
    @device_materialize.plan_numeric_layout(
      plan,
      alignment_bytes=16L,
      max_arena_bytes=127L,
    )
  catch {
    ArenaTooLarge => ()
    _ => fail("unexpected numeric arena error")
  } noraise {
    _ => fail("undersized numeric arena was accepted")
  }
}
