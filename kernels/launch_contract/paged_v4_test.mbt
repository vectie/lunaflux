///|
test "paged v4 admits exact BF16 FP8 and I8 full graphs" {
  for mode in 1..<=3 {
    let plan = launch_v4_plan(mode)
    let inputs = launch_v4_inputs(plan)
    let profiles = [launch_v4_profile()]
    let set = @launch_contract.admit_paged_graph_v4(
      plan,
      launch_v4_catalog(plan),
      launch_v4_layout(plan),
      profiles,
      inputs,
      launch_v4_limits(),
    )
    inputs[0] = inputs[1]
    profiles[0] = try! @launch_contract.PagedKernelProfile::new(
      id=launch_v4_profile().id(),
      max_query_rows=1,
      max_query_tokens=8,
      max_page_table_entries=2,
    )
    assert_eq(set.catalog_version(), @catalog.CatalogVersion::v4())
    assert_eq(set.model_identity(), plan.identity())
    assert_eq(set.numeric_schema_digest(), plan.numeric_binding().digest())
    assert_eq(set.profiles()[0], launch_v4_profile())
    assert_eq(set.contracts().length(), plan.operations().length())
    for index, contract in set.contracts() {
      assert_eq(contract.profile(), launch_v4_profile())
      assert_eq(contract.operation_id(), plan.operations()[index].id())
      assert_eq(
        contract.semantic_version(),
        @catalog.KernelSemanticVersion::v4(),
      )
      assert_eq(
        contract.operation_execution_digest(),
        plan.numeric_binding().operations()[index].execution().digest(),
      )
      assert_eq(contract.entry_point(), launch_v4_entry_point(index))
    }
    let projection = set.contracts()[1]
    let mut scale_count = 0
    for operand in projection.operands() {
      if operand.role() is WeightScaleInput(_, _) {
        scale_count += 1
      }
    }
    assert_eq(scale_count, if mode == 1 { 0 } else { 1 })
    if mode != 1 {
      let weight = try! @plan.TensorRef::new(1)
      guard plan.tensor(weight) is Some(tensor) && tensor.scale() is Some(scale) else {
        fail("scaled fixture lost its schema-owned scale")
      }
      assert_eq(projection.operands()[2].role(), WeightInput(weight))
      assert_eq(
        projection.operands()[3].role(),
        WeightScaleInput(weight, scale),
      )
    }
  }
}

///|
test "paged v4 result lookup is exact and has no subsequence scope" {
  let plan = launch_v4_plan(3)
  let profile = launch_v4_profile()
  let set = @launch_contract.admit_paged_graph_v4(
    plan,
    launch_v4_catalog(plan),
    launch_v4_layout(plan),
    [profile],
    launch_v4_inputs(plan),
    launch_v4_limits(),
  )
  let operation = try! @plan.OperationId::new(1)
  guard set.contract(profile.id(), operation) is Some(contract) else {
    fail("exact v4 contract lookup failed")
  }
  assert_eq(contract.operation_id(), operation)
  assert_eq(contract.profile(), profile)
  assert_eq(contract.kv_layout_version(), 1)
  assert_eq(contract.tokens_per_page(), 4)
}

///|
test "paged v4 input owns its operand table before admission" {
  let plan = launch_v4_plan(1)
  let operands = launch_v4_operands(plan, 0)
  let input = try! @launch_contract.PagedV4AotLaunchContractInput::new(
    profile_id=launch_v4_profile().id(),
    operation_id=try! @plan.OperationId::new(0),
    entry_point=launch_v4_entry_point(0),
    dimensions=dimensions(4),
    operands~,
  )
  operands[0] = operands[1]
  let inputs = launch_v4_inputs(plan)
  inputs[0] = input
  let set = @launch_contract.admit_paged_graph_v4(
    plan,
    launch_v4_catalog(plan),
    launch_v4_layout(plan),
    [launch_v4_profile()],
    inputs,
    launch_v4_limits(),
  )
  assert_eq(set.contracts()[0].operands()[0].role(), RuntimeInput(StepCounts))
}
