///|
fn admit_blueprint_fixture(
  metadata : @spec.LlamaModelMetadata,
  plan : @plan.ModelPlan,
  layout : @device_layout.DeviceKvLayout,
  target : @catalog.DeviceTarget,
  limits : DeviceStepLimits,
  contracts : @launch_contract.PagedKvLaunchContractSet,
  artifacts : @artifact.KernelArtifactBundle,
) -> PagedKvExecutionBlueprint raise {
  admit_paged_kv_execution_blueprint(
    plan,
    limits,
    layout,
    blueprint_memory_plan(metadata, plan, layout, target),
    @launch_contract.KernelProfileId::from_stable_int(1),
    contracts,
    artifacts,
  )
}

///|
test "paged blueprint preserves exact inert semantic and physical bindings" {
  let metadata = blueprint_metadata()
  let plan = blueprint_model_plan(metadata)
  let layout = blueprint_layout(plan)
  let target = blueprint_target()
  let contracts = blueprint_contract_set(plan, layout, target)
  let result = admit_blueprint_fixture(
    metadata,
    plan,
    layout,
    target,
    blueprint_limits(),
    contracts,
    blueprint_artifacts(contracts),
  )
  assert_eq(result.identity(), plan.identity())
  assert_eq(result.target(), target)
  assert_eq(result.catalog_version(), @catalog.CatalogVersion::v3())
  assert_eq(result.profile(), blueprint_profile())
  assert_eq(result.steps().length(), 2)
  let rotary = result.steps()[0]
  assert_eq(rotary.operation_id(), @plan.OperationId::new(3))
  assert_eq(rotary.operation_kind(), RotaryEmbedding)
  assert_eq(rotary.kv_layer(), Some(@plan.KvLayerId::new(0)))
  assert_eq(
    rotary.capability(),
    @plan.KernelCapabilityId::positioned_rotary_embedding(),
  )
  assert_eq(rotary.semantic_version(), @catalog.KernelSemanticVersion::v3())
  assert_eq(rotary.artifact_module_index(), 0)
  assert_eq(rotary.function_symbol(), "paged_exact")
  assert_eq(rotary.operands()[0].space(), StepDescriptor(Counts))
  assert_eq(rotary.operands()[1].space(), StepDescriptor(TokenPositions))
  assert_eq(rotary.operands()[2].space(), ActivationWorkspaceArena)
  assert_eq(rotary.operands()[4].space(), ActivationWorkspaceArena)
  let attention = result.steps()[1]
  assert_eq(attention.operation_id(), @plan.OperationId::new(4))
  assert_eq(
    attention.capability(),
    @plan.KernelCapabilityId::paged_causal_attention(),
  )
  assert_eq(attention.operands()[8].space(), PersistentKvArena)
  assert_eq(attention.operands()[8].offset_bytes(), 0L)
  assert_eq(
    attention.operands()[9].offset_bytes(),
    layout.component_stride_bytes(),
  )
}

///|
test "paged blueprint rejects a full-context arena for a smaller step envelope" {
  let metadata = blueprint_metadata()
  let plan = blueprint_model_plan(metadata)
  let layout = blueprint_layout(plan)
  let target = blueprint_target()
  let contracts = blueprint_contract_set(plan, layout, target)
  try
    admit_paged_kv_execution_blueprint(
      plan,
      blueprint_limits(),
      layout,
      blueprint_memory_plan(metadata, plan, layout, target, full_context=true),
      blueprint_profile().id(),
      contracts,
      blueprint_artifacts(contracts),
    )
  catch {
    InvalidBlueprint(ActivationRegion) => ()
    _ => fail("unexpected activation-envelope mismatch")
  } noraise {
    _ => fail("full-context arena was accepted as exact paged step evidence")
  }
}

///|
test "paged blueprint rejects foreign model identity and device target" {
  let metadata = blueprint_metadata()
  let plan = blueprint_model_plan(metadata)
  let layout = blueprint_layout(plan)
  let target = blueprint_target()
  let contracts = blueprint_contract_set(plan, layout, target)
  let artifacts = blueprint_artifacts(contracts)
  let foreign_metadata = blueprint_metadata(digest_head='f')
  let foreign_plan = blueprint_model_plan(foreign_metadata)
  let foreign_layout = blueprint_layout(foreign_plan)
  let foreign_memory = blueprint_memory_plan(
    foreign_metadata, foreign_plan, foreign_layout, target,
  )
  try
    admit_paged_kv_execution_blueprint(
      plan,
      blueprint_limits(),
      layout,
      foreign_memory,
      @launch_contract.KernelProfileId::from_stable_int(1),
      contracts,
      artifacts,
    )
  catch {
    InvalidBlueprint(Identity) => ()
    _ => fail("unexpected foreign identity error")
  } noraise {
    _ => fail("expected foreign identity rejection")
  }
  let foreign_target = blueprint_target(minor=1)
  let foreign_target_memory = blueprint_memory_plan(
    metadata, plan, layout, foreign_target,
  )
  try
    admit_paged_kv_execution_blueprint(
      plan,
      blueprint_limits(),
      layout,
      foreign_target_memory,
      @launch_contract.KernelProfileId::from_stable_int(1),
      contracts,
      artifacts,
    )
  catch {
    InvalidBlueprint(Target) => ()
    _ => fail("unexpected foreign target error")
  } noraise {
    _ => fail("expected foreign target rejection")
  }
}

///|
test "paged blueprint requires exact profile limits and layout" {
  let metadata = blueprint_metadata()
  let plan = blueprint_model_plan(metadata)
  let layout = blueprint_layout(plan)
  let target = blueprint_target()
  let contracts = blueprint_contract_set(plan, layout, target)
  let artifacts = blueprint_artifacts(contracts)
  let memory = blueprint_memory_plan(metadata, plan, layout, target)
  try
    admit_paged_kv_execution_blueprint(
      plan,
      blueprint_limits(max_tokens=3),
      layout,
      memory,
      @launch_contract.KernelProfileId::from_stable_int(1),
      contracts,
      artifacts,
    )
  catch {
    InvalidBlueprint(Profile) => ()
    _ => fail("unexpected profile-limit error")
  } noraise {
    _ => fail("expected profile-limit rejection")
  }
  try
    admit_paged_kv_execution_blueprint(
      plan,
      blueprint_limits(),
      layout,
      memory,
      @launch_contract.KernelProfileId::from_stable_int(2),
      contracts,
      artifacts,
    )
  catch {
    InvalidBlueprint(Profile) => ()
    _ => fail("unexpected missing profile error")
  } noraise {
    _ => fail("expected missing profile rejection")
  }
  try
    admit_paged_kv_execution_blueprint(
      plan,
      blueprint_limits(),
      blueprint_layout(plan, pages=9),
      memory,
      @launch_contract.KernelProfileId::from_stable_int(1),
      contracts,
      artifacts,
    )
  catch {
    InvalidBlueprint(Layout) => ()
    _ => fail("unexpected foreign layout error")
  } noraise {
    _ => fail("expected foreign layout rejection")
  }
  try
    admit_paged_kv_execution_blueprint(
      plan,
      blueprint_limits(),
      layout,
      memory,
      @launch_contract.KernelProfileId::from_stable_int(1),
      contracts,
      blueprint_v1_artifacts(plan.identity().content(), target),
    )
  catch {
    // A separately authenticated v1 fixture can no longer borrow this plan's
    // digest. Exact identity rejection correctly precedes catalog inspection.
    InvalidBlueprint(Identity) => ()
    _ => fail("unexpected foreign artifact identity error")
  } noraise {
    _ => fail("expected foreign catalog rejection")
  }
}

///|
test "paged blueprint rejects foreign artifact entries and operation graph" {
  let metadata = blueprint_metadata()
  let plan = blueprint_model_plan(metadata)
  let layout = blueprint_layout(plan)
  let target = blueprint_target()
  let contracts = blueprint_contract_set(plan, layout, target)
  let foreign_contracts = blueprint_contract_set(
    plan,
    layout,
    target,
    entry_bias=100,
  )
  try
    admit_blueprint_fixture(
      metadata,
      plan,
      layout,
      target,
      blueprint_limits(),
      contracts,
      blueprint_artifacts(foreign_contracts),
    )
  catch {
    InvalidBlueprintOperation(_, ArtifactEntryPoint) => ()
    _ => fail("unexpected foreign artifact error")
  } noraise {
    _ => fail("expected foreign artifact rejection")
  }
  let reordered = blueprint_model_plan(metadata, extra_projection=true)
  let reordered_layout = blueprint_layout(reordered)
  try
    admit_paged_kv_execution_blueprint(
      reordered,
      blueprint_limits(),
      reordered_layout,
      blueprint_memory_plan(metadata, reordered, reordered_layout, target),
      @launch_contract.KernelProfileId::from_stable_int(1),
      contracts,
      blueprint_artifacts(contracts),
    )
  catch {
    InvalidBlueprint(Identity) => ()
    _ => fail("unexpected operation-order error")
  } noraise {
    _ => fail("expected operation-order rejection")
  }
}

///|
test "paged blueprint rejects same-identity foreign activation liveness graph" {
  let metadata = blueprint_metadata()
  let plan = blueprint_model_plan(metadata)
  let layout = blueprint_layout(plan)
  let target = blueprint_target()
  let contracts = blueprint_contract_set(plan, layout, target)
  let foreign = blueprint_model_plan(metadata, replace_qkv_with_projection=true)
  let foreign_layout = blueprint_layout(foreign)
  try
    admit_paged_kv_execution_blueprint(
      plan,
      blueprint_limits(),
      layout,
      blueprint_memory_plan(metadata, foreign, foreign_layout, target),
      @launch_contract.KernelProfileId::from_stable_int(1),
      contracts,
      blueprint_artifacts(contracts),
    )
  catch {
    InvalidBlueprint(Identity) => ()
    _ => fail("unexpected foreign activation-liveness error")
  } noraise {
    _ => fail("expected foreign activation-liveness rejection")
  }
}

///|
test "paged blueprint rejects same-identity foreign operation shape" {
  let metadata = blueprint_metadata()
  let plan = blueprint_model_plan(metadata)
  let layout = blueprint_layout(plan)
  let target = blueprint_target()
  let contracts = blueprint_contract_set(plan, layout, target)
  let foreign = blueprint_model_plan(metadata, rope_theta=20000.0)
  let foreign_layout = blueprint_layout(foreign)
  try
    admit_paged_kv_execution_blueprint(
      foreign,
      blueprint_limits(),
      foreign_layout,
      blueprint_memory_plan(metadata, foreign, foreign_layout, target),
      @launch_contract.KernelProfileId::from_stable_int(1),
      contracts,
      blueprint_artifacts(contracts),
    )
  catch {
    InvalidBlueprint(Identity) => ()
    _ => fail("unexpected foreign operation-shape error")
  } noraise {
    _ => fail("expected foreign operation-shape rejection")
  }
}

///|
test "paged blueprint rejects activation size alignment and workspace drift" {
  let metadata = blueprint_metadata()
  let plan = blueprint_model_plan(metadata)
  let layout = blueprint_layout(plan)
  let target = blueprint_target()
  let cases = [
    blueprint_contract_set(plan, layout, target, rotary_input_bytes=60L),
    blueprint_contract_set(plan, layout, target, rotary_input_alignment=128L),
  ]
  for index, contracts in cases {
    try
      admit_blueprint_fixture(
        metadata,
        plan,
        layout,
        target,
        blueprint_limits(),
        contracts,
        blueprint_artifacts(contracts),
      )
    catch {
      InvalidBlueprintOperation(_, OperandSize) if index == 0 => ()
      InvalidBlueprintOperation(_, OperandAlignment) if index == 1 => ()
      _ => fail("unexpected operand extent error")
    } noraise {
      _ => fail("expected operand extent rejection")
    }
  }
  let smaller_workspace_plan = blueprint_model_plan(
    metadata,
    rotary_workspace=32L,
  )
  let smaller_workspace_layout = blueprint_layout(smaller_workspace_plan)
  let exact_contracts = blueprint_contract_set(plan, layout, target)
  try
    admit_paged_kv_execution_blueprint(
      smaller_workspace_plan,
      blueprint_limits(),
      smaller_workspace_layout,
      blueprint_memory_plan(
        metadata, smaller_workspace_plan, smaller_workspace_layout, target,
      ),
      @launch_contract.KernelProfileId::from_stable_int(1),
      exact_contracts,
      blueprint_artifacts(exact_contracts),
    )
  catch {
    InvalidBlueprint(Identity) => ()
    _ => fail("unexpected workspace drift error")
  } noraise {
    _ => fail("expected workspace drift rejection")
  }
}
