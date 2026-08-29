///|
fn fp8_v2_simple_fixture() -> Fp8AbiFixture {
  let plan = fixture_plan(projection_workspace=4L)
  let catalog = fixture_catalog(plan)
  let launches = fixture_launches(plan, catalog)
  {
    plan,
    launches,
    kernels: fixture_kernels(plan, catalog, launches),
    weight_scale: try! @fp8_launch_abi.Fp8LaunchStorageRegion::new(
      storage=@fp8_launch_abi.Fp8LaunchStorageIdentity::from_sha256(
        "6666666666666666666666666666666666666666666666666666666666666666",
      ),
      storage_bytes=4L,
      offset=0L,
      byte_length=4L,
      alignment=4L,
    ),
    scale: fixture_region(0L, 4L, 4L),
    workspace: fixture_region(4L, 4L, 4L, workspace=true),
    shape: try! @fp8_launch_abi.Fp8ExactExecutionShape::stateless_full_prefill(
      profile=@launch_contract.KernelProfileId::from_stable_int(1),
      @device_profile.ExactExecutionShape::full_prefill(
        batch_rows=1,
        sequence_tokens=2,
      ),
    ),
  }
}

///|
test "v2 simple FP8 admits one typed whole-Workspace stage deterministically" {
  let fixture = v2_paged_fixture(false)
  let first = try! admit_v2_paged(fixture)
  let repeated = try! admit_v2_paged(fixture)
  assert_eq(first, repeated)
  assert_eq(first.version(), @fp8_launch_abi.Fp8LaunchAbiVersion::v2())
  assert_eq(first.operation_kind(), OutputProjection)
  assert_eq(
    first.activation_scale_policy(),
    @numeric_contract.ActivationScalePolicy::dynamic_per_tensor_f32_v1(),
  )
  assert_eq(first.scale_kind(), SimpleExternalFiniteE4M3F32V2)
  assert_eq(first.launch_source_version(), PagedNumericV4)
  assert_eq(first.launch_dimensions().grid_x(), 2)
  assert_eq(first.raw_operands().length(), 6)
  assert_eq(
    first.raw_operands()[3].role(),
    WeightScaleInput(
      try! @plan.TensorRef::new(1),
      first.weight_scales()[0].scale(),
    ),
  )
  assert_eq(first.raw_operands()[5].role(), Workspace)
  assert_eq(first.workspace(), fixture.workspace)
  assert_eq(first.stages().length(), 1)
  let stage = first.stages()[0]
  assert_eq(stage.ordinal(), 0)
  assert_eq(
    stage.stage(),
    @numeric_contract.ActivationScaleStage::external_operation_input_v1(),
  )
  assert_eq(stage.region(), fixture.workspace)
  let legacy = admit_fixture(fp8_abi_fixture())
  assert_true(first.digest() != legacy.digest())
  inspect(
    first.digest().as_sha256(),
    content="7941b5c14bbe8070791de2dde2bcd45d03d41b1c527ce7d84c29db9629d036ad",
  )
}

///|
test "v2 canonical identity binds exact dimensions and rejects reordered raw operands" {
  let fixture = v2_paged_fixture(false)
  let catalog = v2_paged_catalog(fixture.plan)
  let changed_launches = try! v2_paged_launches(fixture.plan, catalog, grid_x=3)
  let changed_block = try! v2_paged_launches(fixture.plan, catalog, block_x=64)
  let changed_shared = try! v2_paged_launches(
    fixture.plan,
    catalog,
    shared_memory_bytes=32,
  )
  let baseline = try! admit_v2_paged(fixture)
  let changed = try! admit_v2_paged(fixture, launches=changed_launches)
  let block = try! admit_v2_paged(fixture, launches=changed_block)
  let shared = try! admit_v2_paged(fixture, launches=changed_shared)
  assert_eq(baseline.launch_dimensions().grid_x(), 2)
  assert_eq(changed.launch_dimensions().grid_x(), 3)
  assert_true(baseline.digest() != changed.digest())
  assert_true(baseline.digest() != block.digest())
  assert_true(baseline.digest() != shared.digest())

  let reordered = v2_paged_operands(fixture.plan, 1)
  let weight = reordered[2]
  reordered[2] = reordered[3]
  reordered[3] = weight
  try
    v2_paged_launches(fixture.plan, catalog, selected_operands=reordered)
  catch {
    InvalidOperand(operation, PositionalRoleMismatch) =>
      assert_eq(operation.as_int(), 1)
    _ => fail("unexpected reordered FP8 raw operand rejection")
  } noraise {
    _ => fail("reordered FP8 raw operands reached ABI admission")
  }
}

///|
test "v2 simple digest rejects profile replay and binds the exact live shape" {
  let fixture = v2_paged_fixture(false)
  let first = try! admit_v2_paged(fixture)
  let rows = [
    try! @fp8_launch_abi.Fp8PagedRowShape::new(
      query_tokens=1,
      context_tokens=4,
      page_table_entries=1,
    ),
  ]
  let replay = try! @fp8_launch_abi.Fp8ExactExecutionShape::paged_full_graph(
    profile=v2_paged_profile().id(),
    rows~,
  )
  let second = try! admit_v2_paged(fixture, shape=replay)
  assert_true(first.digest() != second.digest())
  rows[0] = try! @fp8_launch_abi.Fp8PagedRowShape::new(
    query_tokens=1,
    context_tokens=8,
    page_table_entries=2,
  )
  assert_eq(second.exact_shape(), replay)
}

///|
test "v2 simple rejects workspace size alignment alias and legacy replay" {
  let fixture = v2_paged_fixture(false)
  let hostile = [
    fixture_region(0L, 8L, 4L, workspace=true),
    fixture_region(8L, 4L, 8L, workspace=true),
  ]
  for workspace in hostile {
    try admit_v2_paged(fixture, workspace~) catch {
      InvalidFp8LaunchAbi(WorkspaceContract) => ()
      _ => fail("unexpected v2 workspace rejection")
    } noraise {
      _ => fail("substituted v2 Workspace was admitted")
    }
  }
  let overlapping = try! @fp8_launch_abi.Fp8LaunchStorageRegion::new(
    storage=fixture.weight_scales[0].storage(),
    storage_bytes=fixture.weight_scales[0].storage_bytes(),
    offset=0L,
    byte_length=4L,
    alignment=4L,
  )
  try admit_v2_paged(fixture, workspace=overlapping) catch {
    InvalidFp8LaunchAbi(RegionAlias) => ()
    _ => fail("unexpected v2 alias rejection")
  } noraise {
    _ => fail("Workspace aliased a weight scale")
  }
  try
    admit_v2_paged(fixture, weight_scales=[fixture_region(0L, 4L, 4L)])
  catch {
    InvalidFp8LaunchAbi(BindingMismatch) => ()
    _ => fail("unexpected raw scale alignment rejection")
  } noraise {
    _ => fail("under-aligned raw scale pointer reached ABI v2")
  }
  let legacy = fp8_v2_simple_fixture()
  try
    @fp8_launch_abi.admit_staged_dynamic_activation_scale_launch_v2(
      legacy.plan,
      legacy.kernels,
      Stateless(legacy.launches),
      operation=try! @plan.OperationId::new(1),
      shape=legacy.shape,
      weight_scale_regions=[legacy.weight_scale],
      workspace=legacy.workspace,
    )
  catch {
    InvalidFp8LaunchAbi(LaunchContract) => ()
    _ => fail("unexpected legacy replay rejection")
  } noraise {
    _ => fail("legacy v1 workspace replayed as ABI v2")
  }
}
