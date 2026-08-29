///|
test "numeric layout facts reject missing extra and foreign ordinals" {
  let plan = d2_wb_plan()
  try
    validate_weight_layout_facts(plan, 16L, 128L, 128L, [
      d2_wb_fact(0, 0L, 64L),
      None,
    ])
  catch {
    MissingLayoutRegion(reference) => assert_eq(reference.as_int(), 1)
    _ => fail("unexpected missing-layout error")
  } noraise {
    _ => fail("missing layout region was accepted")
  }
  try
    validate_weight_layout_facts(plan, 16L, 192L, 192L, [
      d2_wb_fact(0, 0L, 64L),
      d2_wb_fact(1, 64L, 64L),
      d2_wb_fact(2, 128L, 64L),
    ])
  catch {
    InvalidWeightArena(RegionCount) => ()
    _ => fail("unexpected extra-layout error")
  } noraise {
    _ => fail("extra layout region was accepted")
  }
  try
    validate_weight_layout_facts(plan, 16L, 128L, 128L, [
      d2_wb_fact(1, 0L, 64L),
      d2_wb_fact(0, 64L, 64L),
    ])
  catch {
    InvalidLayoutRegion(reference, ReferenceMismatch) =>
      assert_eq(reference.as_int(), 0)
    _ => fail("unexpected foreign-reference error")
  } noraise {
    _ => fail("swapped layout references were accepted")
  }
}

///|
test "numeric layout facts reject wrong length overlap and end overflow" {
  let plan = d2_wb_plan()
  try
    validate_weight_layout_facts(plan, 16L, 128L, 127L, [
      d2_wb_fact(0, 0L, 63L),
      d2_wb_fact(1, 64L, 64L),
    ])
  catch {
    InvalidLayoutRegion(reference, ByteLengthMismatch) =>
      assert_eq(reference.as_int(), 0)
    _ => fail("unexpected byte-length error")
  } noraise {
    _ => fail("wrong numeric byte length was accepted")
  }
  try
    validate_weight_layout_facts(plan, 16L, 128L, 128L, [
      d2_wb_fact(0, 0L, 64L),
      d2_wb_fact(1, 32L, 64L),
    ])
  catch {
    InvalidLayoutRegion(reference, OverlappingPrevious) =>
      assert_eq(reference.as_int(), 1)
    _ => fail("unexpected overlap error")
  } noraise {
    _ => fail("overlapping layout regions were accepted")
  }
  try
    validate_weight_layout_facts(plan, 16L, 9223372036854775807L, 128L, [
      d2_wb_fact(0, 0L, 64L),
      d2_wb_fact(1, 64L, 9223372036854775807L),
    ])
  catch {
    InvalidLayoutRegion(reference, EndOverflow) =>
      assert_eq(reference.as_int(), 1)
    _ => fail("unexpected end-overflow error")
  } noraise {
    _ => fail("overflowing layout end was accepted")
  }
}
