///|
fn graph_bound_operation(
  id : Int,
  input_count : Int,
) -> @plan.PlanOperation raise @plan.PlanValidationError {
  let inputs : Array[@plan.ValueRef] = []
  for _ in 0..<input_count {
    inputs.push(Activation(@plan.ActivationRef::new(0)))
  }
  @plan.PlanOperation::new_stateless(
    id=@plan.OperationId::new(id),
    kind=RmsNorm,
    inputs~,
    outputs=[@plan.ActivationRef::new(id)],
    shape=NormalizationShape(width=1, epsilon=0.000001),
    required_capability=@plan.KernelCapabilityId::rms_norm(),
    workspace_bytes=0L,
  )
}

///|
fn graph_bound_operations(
  count : Int,
) -> ReadOnlyArray[@plan.PlanOperation] raise @plan.PlanValidationError {
  let operations : Array[@plan.PlanOperation] = []
  for id = 0; id < count; id = id + 1 {
    operations.push(graph_bound_operation(id, 64))
  }
  ReadOnlyArray::from_array(operations)
}

///|
test "device graph total-input ceiling matches expanded production plan envelope" {
  validate_graph_bounds(try! graph_bound_operations(256))
  try validate_graph_bounds(graph_bound_operations(257)) catch {
    BoundExceeded(TotalInputCount) => ()
    _ => fail("unexpected device total-input bound failure")
  } noraise {
    _ => fail("device graph admitted more than 16384 total inputs")
  }
}
