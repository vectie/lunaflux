///|
test "all launch input factories bound operand ownership before copying" {
  let operands : Array[@launch_contract.AotOperand] = []
  for _ in 0..<33 {
    operands.push(
      try! @launch_contract.AotOperand::new(
        role=ExternalInput(TokenIds),
        byte_count=4L,
        alignment=4L,
      ),
    )
  }
  let profile_id = try! @launch_contract.KernelProfileId::from_stable_int(1)
  let operation_id = try! @plan.OperationId::new(0)
  let entry_point = entry_point(1, 1)
  let dimensions = dimensions(1)
  for kind in 1..<=3 {
    try {
      match kind {
        1 =>
          ignore(
            @launch_contract.AotLaunchContractInput::new(
              profile_id~,
              operation_id~,
              entry_point~,
              dimensions~,
              operands~,
            ),
          )
        2 =>
          ignore(
            @launch_contract.PagedKvAotLaunchContractInput::new(
              profile_id~,
              operation_id~,
              entry_point~,
              dimensions~,
              operands~,
            ),
          )
        _ =>
          ignore(
            @launch_contract.PagedV4AotLaunchContractInput::new(
              profile_id~,
              operation_id~,
              entry_point~,
              dimensions~,
              operands~,
            ),
          )
      }
    } catch {
      OperandCountExceeded(operation~, maximum~) => {
        assert_eq(operation.as_int(), 0)
        assert_eq(maximum, 32)
      }
      _ => fail("unexpected oversized launch-input error")
    } noraise {
      _ => fail("launch input copied more than the absolute operand bound")
    }
  }
}

///|
test "legacy launch paths explicitly reject weight-scale roles" {
  let scale = @plan.ScaleTensorRef::new(try! @plan.TensorRef::new(1))
  let stateless = embedding_operands(2)
  stateless[1] = try! @launch_contract.AotOperand::new(
    role=WeightScaleInput(try! @plan.TensorRef::new(0), scale),
    byte_count=4L,
    alignment=16L,
  )
  try
    @launch_contract.admit(
      contract_plan(),
      contract_catalog(),
      [profile(1, 2)],
      [
        input(1, 0, 1, 11, 2, stateless),
        input(1, 1, 2, 21, 2, norm_operands(2)),
      ],
      contract_limits(),
    )
  catch {
    InvalidOperand(_, UnexpectedSemanticRole) => ()
    _ => fail("unexpected legacy stateless scale-role error")
  } noraise {
    _ => fail("legacy stateless launch admitted a weight-scale operand")
  }

  let paged = paged_operands(0)
  paged[2] = try! @launch_contract.AotOperand::new(
    role=WeightScaleInput(try! @plan.TensorRef::new(0), scale),
    byte_count=4L,
    alignment=16L,
  )
  let paged_inputs = paged_graph_inputs()
  paged_inputs[0] = try! @launch_contract.PagedKvAotLaunchContractInput::new(
    profile_id=paged_profile_fixture().id(),
    operation_id=try! @plan.OperationId::new(0),
    entry_point=entry_point(11, 1),
    dimensions=dimensions(4),
    operands=paged,
  )
  try
    @launch_contract.admit_paged_graph(
      paged_plan_fixture(),
      paged_catalog_fixture(),
      paged_layout_fixture(),
      [paged_profile_fixture()],
      paged_inputs,
      paged_limits_fixture(),
    )
  catch {
    @launch_contract.InvalidOperand(_, UnexpectedSemanticRole) => ()
    _ => fail("unexpected legacy paged scale-role error")
  } noraise {
    _ => fail("legacy paged launch admitted a weight-scale operand")
  }
}
