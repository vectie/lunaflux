///|
test "numeric layout facts reject leading and interior noncanonical gaps" {
  let plan = d2_wb_plan()
  try
    validate_weight_layout_facts(plan, 16L, 144L, 128L, [
      d2_wb_fact(0, 16L, 64L),
      d2_wb_fact(1, 80L, 64L),
    ])
  catch {
    InvalidLayoutRegion(reference, NonCanonicalOffset) =>
      assert_eq(reference.as_int(), 0)
    _ => fail("unexpected leading-gap error")
  } noraise {
    _ => fail("leading layout gap was accepted")
  }
  try
    validate_weight_layout_facts(plan, 16L, 144L, 128L, [
      d2_wb_fact(0, 0L, 64L),
      d2_wb_fact(1, 80L, 64L),
    ])
  catch {
    InvalidLayoutRegion(reference, NonCanonicalOffset) =>
      assert_eq(reference.as_int(), 1)
    _ => fail("unexpected interior-gap error")
  } noraise {
    _ => fail("interior layout gap was accepted")
  }
}

///|
test "numeric layout facts reject materialized and terminal summaries" {
  let plan = d2_wb_plan()
  try
    validate_weight_layout_facts(plan, 16L, 128L, 127L, d2_wb_exact_facts())
  catch {
    InvalidWeightArena(MaterializedSizeMismatch) => ()
    _ => fail("unexpected materialized-summary error")
  } noraise {
    _ => fail("wrong materialized summary was accepted")
  }
  try
    validate_weight_layout_facts(plan, 16L, 144L, 128L, d2_wb_exact_facts())
  catch {
    InvalidWeightArena(TerminalExtentMismatch) => ()
    _ => fail("unexpected terminal-extent error")
  } noraise {
    _ => fail("wrong terminal extent was accepted")
  }
}
