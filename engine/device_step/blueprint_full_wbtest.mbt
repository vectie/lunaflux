///|
fn full_runtime_bytes(
  input : @plan.OperationRuntimeInput,
  profile : @launch_contract.PagedKernelProfile,
) -> Int64 {
  match input {
    StepCounts => 20L
    QueryPositions => profile.max_query_tokens().to_int64() * 4L
    QueryRowOffsets | PageTableOffsets =>
      (profile.max_query_rows().to_int64() + 1L) * 4L
    SequenceLengths => profile.max_query_rows().to_int64() * 4L
    PageTablePageIndices => profile.max_page_table_entries().to_int64() * 4L
  }
}

///|
fn full_static_plan(
  metadata : @spec.LlamaModelMetadata,
  plan : @plan.ModelPlan,
  layout : @device_layout.DeviceKvLayout,
  target : @catalog.DeviceTarget,
) -> @device_plan.StaticDevicePlan raise {
  let mut rotary_workspace = 0L
  for operation in plan.operations() {
    if operation.kind() == RotaryEmbedding {
      rotary_workspace = operation.workspace_bytes()
    }
  }
  @device_plan.build(
    plan,
    blueprint_weight_layout(metadata),
    blueprint_resolved_catalog(plan, layout, target, rotary_workspace~),
  )
}

///|
fn full_activation_bytes(
  memory : @device_memory.DeviceMemoryPlan,
  reference : @plan.ActivationRef,
  profile : @launch_contract.PagedKernelProfile,
) -> Int64 {
  guard memory.region(reference) is Some(region) else {
    abort("missing fixture activation")
  }
  region.logical_bytes() /
  memory.token_rows() *
  profile.max_query_tokens().to_int64()
}

///|
fn full_contract_operands(
  operation : @plan.PlanOperation,
  static_operation : @device_plan.StaticOperation,
  memory : @device_memory.DeviceMemoryPlan,
  layout : @device_layout.DeviceKvLayout,
  profile : @launch_contract.PagedKernelProfile,
  weight_byte_delta? : Int64 = 0L,
) -> Array[@launch_contract.AotOperand] raise {
  let result : Array[@launch_contract.AotOperand] = []
  for runtime in operation.runtime_inputs() {
    result.push(
      blueprint_operand(
        RuntimeInput(runtime),
        full_runtime_bytes(runtime, profile),
        4L,
      ),
    )
  }
  for index, input in operation.inputs() {
    match input {
      ModelInput(reference) =>
        result.push(
          blueprint_operand(
            ExternalInput(reference),
            profile.max_query_tokens().to_int64() * 4L,
            4L,
          ),
        )
      Tensor(reference) => {
        guard static_operation.inputs()[index] is WeightRegion(region) &&
          region.reference() == reference else {
          abort("missing fixture weight")
        }
        result.push(
          blueprint_operand(
            WeightInput(reference),
            region.byte_length() + weight_byte_delta,
            16L,
          ),
        )
      }
      Activation(reference) =>
        result.push(
          blueprint_operand(
            ActivationInput(reference),
            full_activation_bytes(memory, reference, profile),
            16L,
          ),
        )
    }
  }
  for output in operation.outputs() {
    result.push(
      blueprint_operand(
        ActivationOutput(output),
        full_activation_bytes(memory, output, profile),
        16L,
      ),
    )
  }
  if operation.persistent_state() is KvCacheReadWrite(layer) {
    result.push(
      blueprint_operand(
        PersistentKvComponent(layer, Key),
        layout.component_stride_bytes(),
        layout.segment_alignment_bytes(),
      ),
    )
    result.push(
      blueprint_operand(
        PersistentKvComponent(layer, Value),
        layout.component_stride_bytes(),
        layout.segment_alignment_bytes(),
      ),
    )
  }
  if static_operation.binding().workspace_bytes() > 0L {
    result.push(
      blueprint_operand(
        Workspace,
        static_operation.binding().workspace_bytes(),
        static_operation.binding().workspace_alignment(),
      ),
    )
  }
  result
}

///|
fn full_contract_set(
  plan : @plan.ModelPlan,
  static_plan : @device_plan.StaticDevicePlan,
  memory : @device_memory.DeviceMemoryPlan,
  layout : @device_layout.DeviceKvLayout,
  target : @catalog.DeviceTarget,
  weight_byte_delta? : Int64 = 0L,
  entry_bias? : Int = 0,
) -> @launch_contract.PagedKvLaunchContractSet raise {
  let profile = blueprint_profile()
  let inputs : Array[@launch_contract.PagedKvAotLaunchContractInput] = []
  let digest = @catalog.AotArtifactDigest::from_sha256(
    "039058c6f2c0cb492c533b0a4d14ef77cc0f78abccced5287d84a1a2011cfb81",
  )
  for index, operation in plan.operations() {
    let family = @catalog.AotKernelFamily::new(
      module_digest=digest,
      family_id=@catalog.AotKernelFamilyId::from_stable_int(index + 1),
    )
    inputs.push(
      @launch_contract.PagedKvAotLaunchContractInput::new(
        profile_id=profile.id(),
        operation_id=operation.id(),
        entry_point=@catalog.AotKernelEntryPoint::new(
          family~,
          entry_point_id=@catalog.AotKernelEntryPointId::from_stable_int(
            index + entry_bias + 1,
          ),
        ),
        dimensions=blueprint_dimensions(),
        operands=full_contract_operands(
          operation,
          static_plan.operations()[index],
          memory,
          layout,
          profile,
          weight_byte_delta~,
        ),
      ),
    )
  }
  @launch_contract.admit_paged_graph(
    plan,
    blueprint_resolved_catalog(plan, layout, target),
    layout,
    [profile],
    inputs,
    @launch_contract.LaunchContractLimits::new(
      max_profiles=4,
      max_contracts=16,
      max_operands_per_contract=16,
    ),
  )
}

///|
test "full paged blueprint rejects contract weight-region drift" {
  let metadata = blueprint_metadata()
  let plan = blueprint_model_plan(metadata)
  let layout = blueprint_layout(plan)
  let target = blueprint_target()
  let static_plan = full_static_plan(metadata, plan, layout, target)
  let memory = blueprint_memory_plan(metadata, plan, layout, target)
  let contracts = full_contract_set(
    plan,
    static_plan,
    memory,
    layout,
    target,
    weight_byte_delta=2L,
  )
  try
    admit_paged_graph_execution_blueprint(
      plan,
      blueprint_limits(),
      layout,
      static_plan,
      memory,
      blueprint_profile().id(),
      contracts,
      blueprint_artifacts(contracts),
    )
  catch {
    InvalidBlueprintOperation(operation, OperandSize) =>
      assert_eq(operation.as_int(), 0)
    _ => fail("unexpected full-graph weight drift error")
  } noraise {
    _ => fail("expected full-graph weight drift rejection")
  }
}

///|
test "full paged blueprint binds every physical operand space" {
  let metadata = blueprint_metadata()
  let plan = blueprint_model_plan(metadata)
  let layout = blueprint_layout(plan)
  let target = blueprint_target()
  let static_plan = full_static_plan(metadata, plan, layout, target)
  let memory = blueprint_memory_plan(metadata, plan, layout, target)
  let contracts = full_contract_set(plan, static_plan, memory, layout, target)
  let artifacts = blueprint_artifacts(contracts)
  let blueprint = admit_paged_graph_execution_blueprint(
    plan,
    blueprint_limits(),
    layout,
    static_plan,
    memory,
    blueprint_profile().id(),
    contracts,
    artifacts,
  )
  assert_eq(blueprint.scope(), FullGraph)
  assert_eq(blueprint.steps().length(), plan.operations().length())
  assert_eq(blueprint.weight_arena_bytes(), static_plan.weight_arena_bytes())
  assert_eq(blueprint.weight_alignment(), static_plan.weight_alignment())
  assert_eq(blueprint.activation_workspace_arena_bytes(), memory.arena_bytes())
  let embedding = blueprint.steps()[0].operands()
  assert_eq(embedding[0].space(), StepDescriptor(Counts))
  assert_eq(embedding[1].space(), StepDescriptor(TokenIds))
  assert_eq(embedding[2].space(), WeightArena)
  assert_eq(embedding[3].space(), ActivationWorkspaceArena)
  let attention = blueprint.steps().last().unwrap().operands()
  assert_eq(attention[attention.length() - 2].space(), PersistentKvArena)
  assert_eq(attention[attention.length() - 1].space(), PersistentKvArena)
}

///|
test "narrow paged blueprint rejects full-graph contract scope" {
  let metadata = blueprint_metadata()
  let plan = blueprint_model_plan(metadata)
  let layout = blueprint_layout(plan)
  let target = blueprint_target()
  let static_plan = full_static_plan(metadata, plan, layout, target)
  let memory = blueprint_memory_plan(metadata, plan, layout, target)
  let contracts = full_contract_set(plan, static_plan, memory, layout, target)
  try
    admit_paged_kv_execution_blueprint(
      plan,
      blueprint_limits(),
      layout,
      memory,
      blueprint_profile().id(),
      contracts,
      blueprint_artifacts(contracts),
    )
  catch {
    InvalidBlueprint(ContractSemantics) => ()
    _ => fail("unexpected full-scope rejection")
  } noraise {
    _ => fail("expected narrow blueprint scope rejection")
  }
}
