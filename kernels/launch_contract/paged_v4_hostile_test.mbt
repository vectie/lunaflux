///|
fn expect_v4_operand_rejection(
  plan : @plan.ModelPlan,
  operands : Array[@launch_contract.AotOperand],
  expected : @launch_contract.OperandRule,
  operation_index? : Int = 1,
) -> Unit raise {
  try
    @launch_contract.admit_paged_graph_v4(
      plan,
      launch_v4_catalog(plan),
      launch_v4_layout(plan),
      [launch_v4_profile()],
      launch_v4_inputs(
        plan,
        override_operation=operation_index,
        override_operands=operands,
      ),
      launch_v4_limits(),
    )
  catch {
    InvalidOperand(operation, actual) if actual == expected =>
      assert_eq(operation.as_int(), operation_index)
    _ => fail("unexpected v4 operand rejection")
  } noraise {
    _ => fail("hostile v4 operand table was admitted")
  }
}

///|
test "paged v4 rejects missing extra reordered and swapped scales" {
  let i8 = launch_v4_plan(3)
  let exact = launch_v4_operands(i8, 1)
  let missing = exact.copy()
  ignore(missing.remove(3))
  expect_v4_operand_rejection(i8, missing, MissingSemanticRole)

  let reordered = exact.copy()
  reordered[2] = exact[3]
  reordered[3] = exact[2]
  expect_v4_operand_rejection(i8, reordered, PositionalRoleMismatch)

  let swapped = exact.copy()
  let weight = try! @plan.TensorRef::new(1)
  let foreign_scale = @plan.ScaleTensorRef::new(try! @plan.TensorRef::new(0))
  swapped[3] = launch_v4_operand(
    WeightScaleInput(weight, foreign_scale),
    exact[3].byte_count(),
    16L,
  )
  expect_v4_operand_rejection(i8, swapped, PositionalRoleMismatch)

  let bf16 = launch_v4_plan(1)
  let extra = launch_v4_operands(bf16, 1)
  extra.insert(
    3,
    launch_v4_operand(WeightScaleInput(weight, foreign_scale), 4L, 16L),
  )
  expect_v4_operand_rejection(bf16, extra, MissingSemanticRole)
}

///|
test "paged v4 rejects wrong numeric byte counts and alignment" {
  let plan = launch_v4_plan(3)
  let bytes = launch_v4_operands(plan, 1)
  bytes[3] = launch_v4_operand(bytes[3].role(), bytes[3].byte_count() + 1L, 16L)
  expect_v4_operand_rejection(plan, bytes, NumericByteCountMismatch)

  let alignment = launch_v4_operands(plan, 1)
  alignment[2] = launch_v4_operand(
    alignment[2].role(),
    alignment[2].byte_count(),
    32L,
  )
  expect_v4_operand_rejection(plan, alignment, NumericByteCountMismatch)
}

///|
test "paged v4 requires exact canonical full-graph input order and count" {
  let plan = launch_v4_plan(3)
  try
    @launch_contract.admit_paged_graph_v4(
      plan,
      launch_v4_catalog(plan),
      launch_v4_layout(plan),
      [launch_v4_profile()],
      launch_v4_inputs(plan, omit_operation=3),
      launch_v4_limits(),
    )
  catch {
    ContractCountMismatch => ()
    _ => fail("unexpected v4 contract-count error")
  } noraise {
    _ => fail("incomplete v4 graph was admitted")
  }

  let reordered = launch_v4_inputs(plan)
  let first = reordered[0]
  reordered[0] = reordered[1]
  reordered[1] = first
  try
    @launch_contract.admit_paged_graph_v4(
      plan,
      launch_v4_catalog(plan),
      launch_v4_layout(plan),
      [launch_v4_profile()],
      reordered,
      launch_v4_limits(),
    )
  catch {
    BindingOrderMismatch(_) => ()
    _ => fail("unexpected v4 contract-order error")
  } noraise {
    _ => fail("reordered v4 graph was admitted")
  }
}

///|
test "paged v4 requires contiguous declaration-order profile IDs" {
  let plan = launch_v4_plan(1)
  let profile = try! @launch_contract.PagedKernelProfile::new(
    id=@launch_contract.KernelProfileId::from_stable_int(2),
    max_query_rows=1,
    max_query_tokens=4,
    max_page_table_entries=2,
  )
  try
    @launch_contract.admit_paged_graph_v4(
      plan,
      launch_v4_catalog(plan),
      launch_v4_layout(plan),
      [profile],
      launch_v4_inputs(plan),
      launch_v4_limits(),
    )
  catch {
    InvalidProfile(OutsideModelEnvelope) => ()
    _ => fail("unexpected v4 profile-order error")
  } noraise {
    _ => fail("noncanonical v4 profile ID was admitted")
  }
}

///|
test "paged v4 requires the exact catalog-owned entry point" {
  let plan = launch_v4_plan(1)
  let inputs = launch_v4_inputs(plan)
  inputs[1] = try! @launch_contract.PagedV4AotLaunchContractInput::new(
    profile_id=launch_v4_profile().id(),
    operation_id=try! @plan.OperationId::new(1),
    entry_point=launch_v4_entry_point(0),
    dimensions=dimensions(4),
    operands=launch_v4_operands(plan, 1),
  )
  try
    @launch_contract.admit_paged_graph_v4(
      plan,
      launch_v4_catalog(plan),
      launch_v4_layout(plan),
      [launch_v4_profile()],
      inputs,
      launch_v4_limits(),
    )
  catch {
    EntryPointMismatch(_, operation) => assert_eq(operation.as_int(), 1)
    _ => fail("unexpected v4 entry-point error")
  } noraise {
    _ => fail("foreign v4 entry point was admitted")
  }
}

///|
test "paged v4 cross-authenticates model KV and operation numeric identities" {
  let exact = launch_v4_plan(3)
  let changed = launch_v4_plan(
    3,
    content_digest="1111111111111111111111111111111111111111111111111111111111111111",
  )
  assert_eq(exact.identity().plan(), changed.identity().plan())
  assert_false(exact.identity() == changed.identity())
  try
    @launch_contract.admit_paged_graph_v4(
      changed,
      launch_v4_catalog(exact),
      launch_v4_layout(exact),
      [launch_v4_profile()],
      launch_v4_inputs(changed),
      launch_v4_limits(),
    )
  catch {
    IdentityMismatch => ()
    _ => fail("unexpected foreign full-model identity error")
  } noraise {
    _ => fail("foreign full ModelIdentity was admitted")
  }

  let foreign_geometry = try! @plan.KvCacheGeometry::new(
    layer_count=1,
    kv_head_count=1,
    head_dimension=2,
    dtype=BFloat16,
    layout=LayerMajorSplitKeyValueV1,
  )
  let foreign_layout = try! @device_layout.plan(
    exact.identity(),
    foreign_geometry,
    @device_layout.DeviceKvLimits::new(
      tokens_per_page=4,
      total_page_count=4,
      max_arena_bytes=4096L,
    ),
  )
  try
    @launch_contract.admit_paged_graph_v4(
      exact,
      launch_v4_catalog(exact),
      foreign_layout,
      [launch_v4_profile()],
      launch_v4_inputs(exact),
      launch_v4_limits(),
    )
  catch {
    DeviceKvLayoutMismatch => ()
    _ => fail("unexpected foreign KV geometry error")
  } noraise {
    _ => fail("foreign KV geometry was admitted")
  }

  let fp8 = launch_v4_plan(2)
  assert_false(exact.identity().plan() == fp8.identity().plan())
  try
    @launch_contract.admit_paged_graph_v4(
      exact,
      launch_v4_catalog(fp8),
      launch_v4_layout(exact),
      [launch_v4_profile()],
      launch_v4_inputs(exact),
      launch_v4_limits(),
    )
  catch {
    IdentityMismatch => ()
    _ => fail("unexpected foreign operation numeric digest error")
  } noraise {
    _ => fail("foreign operation execution digest was admitted")
  }
}

///|
test "paged v4 rejects legacy catalog before contract validation" {
  let plan = launch_v4_plan(1)
  try
    @launch_contract.admit_paged_graph_v4(
      plan,
      paged_catalog_fixture(),
      launch_v4_layout(plan),
      [launch_v4_profile()],
      [],
      launch_v4_limits(),
    )
  catch {
    @launch_contract.CatalogVersionMismatch => ()
    _ => fail("unexpected legacy catalog error")
  } noraise {
    _ => fail("legacy catalog reached v4 contract validation")
  }
}

///|
test "paged v4 revalidates runtime activation external and KV byte rules" {
  let plan = launch_v4_plan(3)
  let runtime = launch_v4_operands(plan, 3)
  runtime[0] = launch_v4_operand(runtime[0].role(), 24L, 4L)
  expect_v4_operand_rejection(
    plan,
    runtime,
    RuntimeInputSizeMismatch,
    operation_index=3,
  )

  let activation = launch_v4_operands(plan, 1)
  activation[1] = launch_v4_operand(activation[1].role(), 10L, 16L)
  expect_v4_operand_rejection(plan, activation, NumericByteCountMismatch)

  let external = launch_v4_operands(plan, 0)
  external[1] = launch_v4_operand(external[1].role(), 12L, 4L)
  try
    @launch_contract.admit_paged_graph_v4(
      plan,
      launch_v4_catalog(plan),
      launch_v4_layout(plan),
      [launch_v4_profile()],
      launch_v4_inputs(plan, override_operation=0, override_operands=external),
      launch_v4_limits(),
    )
  catch {
    InvalidOperand(operation, NumericByteCountMismatch) =>
      assert_eq(operation.as_int(), 0)
    _ => fail("unexpected external-input byte error")
  } noraise {
    _ => fail("wrong external-input bytes were admitted")
  }

  let kv = launch_v4_operands(plan, 3)
  kv[8] = launch_v4_operand(
    kv[8].role(),
    kv[8].byte_count() + 1L,
    kv[8].alignment(),
  )
  try
    @launch_contract.admit_paged_graph_v4(
      plan,
      launch_v4_catalog(plan),
      launch_v4_layout(plan),
      [launch_v4_profile()],
      launch_v4_inputs(plan, override_operation=3, override_operands=kv),
      launch_v4_limits(),
    )
  catch {
    InvalidOperand(operation, PersistentKvMismatch) =>
      assert_eq(operation.as_int(), 3)
    _ => fail("unexpected persistent-KV byte error")
  } noraise {
    _ => fail("wrong persistent-KV bytes were admitted")
  }
}

///|
test "paged v4 bounds profile operation product before contract allocation" {
  let plan = launch_v4_plan(1)
  let limits = try! @launch_contract.LaunchContractLimits::new(
    max_profiles=1,
    max_contracts=3,
    max_operands_per_contract=20,
  )
  try
    @launch_contract.admit_paged_graph_v4(
      plan,
      launch_v4_catalog(plan),
      launch_v4_layout(plan),
      [launch_v4_profile()],
      launch_v4_inputs(plan),
      limits,
    )
  catch {
    ContractCountExceeded(maximum~) => assert_eq(maximum, 3)
    _ => fail("unexpected v4 product-bound error")
  } noraise {
    _ => fail("v4 exceeded the checked contract product bound")
  }
}
