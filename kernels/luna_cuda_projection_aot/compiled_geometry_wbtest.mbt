///|
test "manifest validation consumes compiled geometry and still rejects a foreign launch" {
  let plan = projection_fixture_plan()
  let operation = plan.operations()[1]
  let contracts = projection_fixture_contracts(qkv_grid_x=2)
  let contract = contracts.contracts()[1]
  let parameters = projection_bind_qkv_widths(
    operation,
    projection_parameters(operation),
    contract.operands(),
  )
  let dispatch = projection_build_runtime_dispatch(
    parameters,
    contract.profile().max_query_tokens(),
    projection_fixture_target(),
    [],
  ).dispatch_plan(contract.profile().max_query_tokens())
  let untuned = projection_dimensions(contract.profile(), parameters, dispatch)
  assert_true(untuned != contract.dimensions())
  // The private, source-bound lowering supplies this geometry. A semantic
  // strategy is not an alternative source of launch authority at binding time.
  validate_projection_contract(
    operation,
    contract,
    dispatch,
    contract.dimensions(),
  )
  try
    validate_projection_contract(operation, contract, dispatch, untuned)
  catch {
    Mismatch(LaunchDimensions) => ()
    _ => fail("wrong launch mismatch")
  } noraise {
    _ => fail("foreign compiled geometry accepted")
  }
}
