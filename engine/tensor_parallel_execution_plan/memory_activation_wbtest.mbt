///|
test "activation consumer byte mismatch is rejected including final output" {
  let facts : Array[MutableActivationFact] = [
    {
      reference: @plan.ActivationRef::new(0),
      byte_count: 16L,
      alignment: 16L,
      producer: @plan.OperationId::new(0),
      live_through: 0,
      slot: -1,
    },
  ]
  try
    record_activation_input(facts, @plan.ActivationRef::new(0), 8L, 1, 0)
  catch {
    InvalidTensorParallelExecutionPlan(rule=ActivationRegion, ..) => ()
    _ => fail("unexpected activation-size rejection")
  } noraise {
    _ => fail("final-output consumer size mismatch was admitted")
  }
}
