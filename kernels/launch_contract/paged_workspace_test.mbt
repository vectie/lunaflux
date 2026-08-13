///|
test "catalog workspace controls the final v3 operand position" {
  let rotary = paged_operands(1)
  try
    @launch_contract.admit_paged_kv(
      paged_plan_fixture(),
      paged_catalog_fixture(rotary_workspace=64L),
      paged_layout_fixture(),
      [paged_profile_fixture()],
      [
        @launch_contract.PagedKvAotLaunchContractInput::new(
          profile_id=@launch_contract.KernelProfileId::from_stable_int(1),
          operation_id=@plan.OperationId::new(1),
          entry_point=entry_point(12, 2),
          dimensions=dimensions(4),
          operands=rotary,
        ),
      ],
      paged_limits_fixture(),
    )
  catch {
    @launch_contract.InvalidOperand(id, MissingSemanticRole) =>
      assert_eq(id.as_int(), 1)
    _ => fail("unexpected catalog-workspace error")
  } noraise {
    _ => fail("expected catalog workspace operand requirement")
  }
}
