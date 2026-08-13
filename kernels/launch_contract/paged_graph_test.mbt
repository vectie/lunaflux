///|
fn paged_graph_inputs(
  omit_operation? : Int = -1,
) -> Array[@launch_contract.PagedKvAotLaunchContractInput] raise {
  let profile_id = @launch_contract.KernelProfileId::from_stable_int(1)
  let inputs : Array[@launch_contract.PagedKvAotLaunchContractInput] = []
  for operation in 0..<3 {
    if operation != omit_operation {
      inputs.push(
        @launch_contract.PagedKvAotLaunchContractInput::new(
          profile_id~,
          operation_id=@plan.OperationId::new(operation),
          entry_point=entry_point(operation + 11, operation + 1),
          dimensions=dimensions(4),
          operands=paged_operands(operation),
        ),
      )
    }
  }
  inputs
}

///|
test "full paged graph admission is exact and explicitly scoped" {
  let profile = paged_profile_fixture()
  let set = @launch_contract.admit_paged_graph(
    paged_plan_fixture(),
    paged_catalog_fixture(),
    paged_layout_fixture(),
    [profile],
    paged_graph_inputs(),
    paged_limits_fixture(),
  )
  assert_eq(set.scope(), FullGraph)
  assert_eq(set.contracts().length(), 3)
  for operation in 0..<3 {
    let contract = set.contracts()[operation]
    assert_eq(contract.profile().id(), profile.id())
    assert_eq(contract.operation_id().as_int(), operation)
    assert_eq(contract.operands()[0].role(), RuntimeInput(StepCounts))
  }
}

///|
test "full paged graph admission rejects every missing operation" {
  for missing in 0..<3 {
    try
      @launch_contract.admit_paged_graph(
        paged_plan_fixture(),
        paged_catalog_fixture(),
        paged_layout_fixture(),
        [paged_profile_fixture()],
        paged_graph_inputs(omit_operation=missing),
        paged_limits_fixture(),
      )
    catch {
      @launch_contract.MissingContract(_, operation) =>
        assert_eq(operation.as_int(), missing)
      _ => fail("unexpected missing full-graph contract failure")
    } noraise {
      _ => fail("expected missing full-graph contract rejection")
    }
  }
}
