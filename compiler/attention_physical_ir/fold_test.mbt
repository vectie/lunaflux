///|
test "online fold has explicit loop-carried SSA dependencies" {
  let plan = @attention_physical_ir.OnlineFold::new(
    queries=16,
    keys=64,
    dimensions=128,
  ).unwrap()
  let defined : Array[@attention_physical_ir.FoldValue] = [
    QueryOperand,
    KeyOperand,
    ValueOperand,
    VisibilityControl,
    ScoreScale,
    PreviousMaximum,
    PreviousDenominator,
    PreviousOutput,
  ]
  for operations in [plan.score_phase(), plan.value_phase()] {
    for operation in operations {
      for value in operation.reads() {
        assert_true(defined.contains(value))
      }
      for value in operation.writes() {
        assert_false(defined.contains(value))
        defined.push(value)
      }
    }
  }
  assert_eq(plan.carried_state().length(), 3)
  for (next, previous) in plan.carried_state() {
    assert_true(defined.contains(next))
    assert_true(defined.contains(previous))
    assert_false(next == previous)
  }
  assert_eq(plan.value_phase(), [
    PrepareScores,
    MergeMaximum,
    RescaleOutput,
    AccumulateProbabilityValue,
    UpdateDenominator,
  ])
}

///|
test "score phase may complete before value readiness but never reads V" {
  let plan = @attention_physical_ir.OnlineFold::new(
    queries=32,
    keys=128,
    dimensions=64,
  ).unwrap()
  for operation in plan.score_phase() {
    assert_false(operation.reads().contains(ValueOperand))
  }
  assert_true(
    @attention_physical_ir.FoldOperation::AccumulateProbabilityValue
    .reads()
    .contains(ValueOperand),
  )
  assert_true(
    @attention_physical_ir.FoldOperation::UpdateDenominator
    .reads()
    .contains(ProbabilitySum),
  )
  assert_false(
    @attention_physical_ir.FoldOperation::UpdateDenominator
    .reads()
    .contains(NextOutput),
  )
}

///|
test "online fold rejects invalid geometry without allocating a tile" {
  for
    (q, k, d) in [
      (0, 16, 16),
      (16, -1, 16),
      (16, 16, 0),
      (2147483647, 2, 1),
      (2, 1, 2147483647),
    ] {
    assert_true(
      @attention_physical_ir.OnlineFold::new(queries=q, keys=k, dimensions=d)
      is None,
    )
  }
  let large = @attention_physical_ir.OnlineFold::new(
    queries=1,
    keys=2147483647,
    dimensions=2147483647,
  ).unwrap()
  assert_eq(large.geometry(), (1, 2147483647, 2147483647))
  assert_eq(large.score_phase().length(), 1)
}

///|
test "canonical fold identity records geometry and floating point contract" {
  let make = k => {
    @attention_physical_ir.OnlineFold::new(queries=16, keys=k, dimensions=128).unwrap()
  }
  assert_eq(make(64).canonical(), make(64).canonical())
  assert_false(make(64).canonical() == make(128).canonical())
  assert_true(make(64).canonical().contains("denominator=unrounded-f32"))
  assert_true(make(64).canonical().contains("probability=bf16-rne"))
}

///|
test "approximate exponential law is immutable and distinct from strict fold" {
  let strict = @attention_physical_ir.OnlineFold::new(
    queries=16,
    keys=64,
    dimensions=128,
  ).unwrap()
  let approximate = strict.with_exponential_law(ApproximateBaseTwoExponential)
  assert_eq(strict.exponential_law(), StrictNaturalExponential)
  assert_eq(approximate.exponential_law(), ApproximateBaseTwoExponential)
  assert_eq(strict.geometry(), approximate.geometry())
  assert_eq(strict.value_phase(), approximate.value_phase())
  assert_false(strict.canonical() == approximate.canonical())
  assert_true(
    approximate.canonical().contains("exponential=approx-base2-f32-v1"),
  )
  assert_true(approximate.canonical().contains("subnormal=flush-permitted"))
}
