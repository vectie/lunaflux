///|
test "v2 gated FP8 derives two ordered cells from one whole Workspace" {
  let fixture = v2_paged_fixture(true)
  let abi = try! admit_v2_paged(fixture)
  assert_eq(abi.version(), @fp8_launch_abi.Fp8LaunchAbiVersion::v2())
  assert_eq(abi.operation_kind(), GatedMlp)
  assert_eq(
    abi.activation_scale_policy(),
    @numeric_contract.ActivationScalePolicy::gated_mlp_external_and_post_silu_bound_expf_f32_v2(),
  )
  assert_eq(abi.scale_kind(), GatedExternalAndPostSiluFiniteE4M3F32V2)
  assert_eq(abi.launch_source_version(), PagedNumericV4)
  assert_eq(abi.raw_operands().length(), 10)
  assert_eq(abi.raw_operands()[9].role(), Workspace)
  for index in 0..<3 {
    assert_eq(
      abi.raw_operands()[3 + index * 2].role(),
      WeightScaleInput(
        try! @plan.TensorRef::new(index + 1),
        abi.weight_scales()[index].scale(),
      ),
    )
  }
  assert_eq(abi.workspace(), fixture.workspace)
  assert_eq(abi.stages().length(), 2)
  assert_eq(abi.stages()[0].ordinal(), 0)
  assert_eq(abi.stages()[1].ordinal(), 1)
  assert_eq(
    abi.stages()[0].stage(),
    @numeric_contract.ActivationScaleStage::external_operation_input_v1(),
  )
  assert_eq(
    abi.stages()[1].stage(),
    @numeric_contract.ActivationScaleStage::post_silu_gate_up_product_v1(),
  )
  assert_eq(abi.stages()[0].region().offset(), fixture.workspace.offset())
  assert_eq(abi.stages()[1].region().offset(), fixture.workspace.offset() + 4L)
  for stage in abi.stages() {
    assert_eq(stage.region().storage(), fixture.workspace.storage())
    assert_eq(stage.region().storage_bytes(), fixture.workspace.storage_bytes())
    assert_eq(stage.region().byte_length(), 4L)
    assert_eq(stage.region().alignment(), 4L)
  }
  inspect(
    abi.digest().as_sha256(),
    content="939d1eab58263fe2a6202714d4dac9b0d337fdc78e4a8ae986feebc7c5f46352",
  )
}

///|
test "v2 gated rejects simple-policy legacy kind and mixed Workspace replay" {
  let legacy = v2_gated_fixture(compound=false, workspace=64L)
  let scales = v2_gated_weight_scales()
  let hostile_workspace = fixture_region(16L, 8L, 4L, workspace=true)
  try
    @fp8_launch_abi.admit_staged_dynamic_activation_scale_launch_v2(
      legacy.plan,
      legacy.kernels,
      Stateless(legacy.launches),
      operation=try! @plan.OperationId::new(1),
      shape=legacy.shape,
      weight_scale_regions=scales,
      workspace=hostile_workspace,
    )
  catch {
    InvalidFp8LaunchAbi(OperationKind) => ()
    _ => fail("unexpected legacy gated policy rejection")
  } noraise {
    _ => fail("simple-policy GatedMlp replayed as ABI v2")
  }
  let gated = v2_paged_fixture(true)
  let simple_workspace = fixture_region(16L, 4L, 4L, workspace=true)
  try admit_v2_paged(gated, workspace=simple_workspace) catch {
    InvalidFp8LaunchAbi(WorkspaceContract) => ()
    _ => fail("unexpected mixed Workspace rejection")
  } noraise {
    _ => fail("one-stage Workspace replayed for compound GatedMlp")
  }
}

///|
test "v2 gated rejects legacy stateless source despite valid compound policy" {
  let legacy = v2_gated_fixture()
  try
    @fp8_launch_abi.admit_staged_dynamic_activation_scale_launch_v2(
      legacy.plan,
      legacy.kernels,
      Stateless(legacy.launches),
      operation=try! @plan.OperationId::new(1),
      shape=legacy.shape,
      weight_scale_regions=v2_gated_weight_scales(),
      workspace=legacy.workspace,
    )
  catch {
    InvalidFp8LaunchAbi(LaunchContract) => ()
    _ => fail("unexpected legacy compound source rejection")
  } noraise {
    _ => fail("legacy stateless compound source reached ABI v2")
  }
}
