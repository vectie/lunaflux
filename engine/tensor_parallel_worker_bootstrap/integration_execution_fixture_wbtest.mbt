///|
fn tp_execution_family_index(operation : @plan.PlanOperation) -> Int {
  match operation.kind() {
    Embedding => 0
    RmsNorm => 1
    QkvProjection => 2
    RotaryEmbedding => 3
    CausalAttention => 4
    OutputProjection => 5
    ResidualAdd => 6
    GatedMlp => 8
    LanguageModelHead => 11
  }
}

///|
fn tp_execution_family(
  operation : @plan.PlanOperation,
  digest : @catalog.AotArtifactDigest,
) -> @catalog.AotKernelFamily raise {
  @catalog.AotKernelFamily::new(
    module_digest=digest,
    family_id=@catalog.AotKernelFamilyId::from_stable_int(
      tp_execution_family_index(operation) + 1,
    ),
  )
}

///|
fn tp_execution_semantics(
  operation : @plan.PlanOperation,
) -> @catalog.KernelExecutionSemantics raise {
  let kv_layout = match operation.persistent_state() {
    NoPersistentState => None
    KvCacheReadWrite(_) =>
      Some(
        @catalog.PagedKvLayoutContract::new(
          layout=LayerMajorSplitKeyValueV1,
          stable_version=1,
          tokens_per_page=128,
        ),
      )
  }
  @catalog.KernelExecutionSemantics::paged_v3(
    kv_layer=operation.kv_layer(),
    runtime_inputs=operation.runtime_inputs(),
    persistent_state=operation.persistent_state(),
    kv_layout~,
  )
}

///|
fn tp_execution_full_layout(
  model : @plan.ModelPlan,
) -> @device_layout.DeviceKvLayout raise {
  @device_layout.plan(
    model.identity(),
    model.kv_cache_geometry(),
    @device_layout.DeviceKvLimits::new(
      tokens_per_page=128,
      total_page_count=4,
      max_arena_bytes=1048576L,
    ),
  )
}

///|
fn tp_execution_catalog(
  model : @plan.ModelPlan,
  target : @catalog.DeviceTarget,
  digest : @catalog.AotArtifactDigest,
) -> @catalog.ResolvedKernelCatalog raise {
  let entries : Array[@catalog.CatalogEntry] = []
  for index, operation in model.operations() {
    if index != tp_execution_family_index(operation) {
      continue
    }
    entries.push(
      @catalog.CatalogEntry::new_paged_v3(
        operation_kind=operation.kind(),
        operation_shape=operation.shape(),
        capability=operation.required_capability(),
        execution_semantics=tp_execution_semantics(operation),
        compute_major=target.compute_major(),
        compute_minor=target.compute_minor(),
        implementation=ContentAddressedAot(
          tp_execution_family(operation, digest),
        ),
        workspace_bytes=operation.workspace_bytes(),
        workspace_alignment=if operation.workspace_bytes() == 0L {
          1L
        } else {
          16L
        },
      ),
    )
  }
  @catalog.KernelCatalog::new(version=@catalog.CatalogVersion::v3(), entries~).resolve_paged(
    model,
    tp_execution_full_layout(model),
    target,
  )
}

///|
fn tp_execution_profile() -> @launch_contract.PagedKernelProfile raise {
  @launch_contract.PagedKernelProfile::new(
    id=@launch_contract.KernelProfileId::from_stable_int(1),
    max_query_rows=2,
    max_query_tokens=16,
    max_page_table_entries=2,
  )
}

///|
fn tp_execution_dimensions() -> @launch_contract.AotLaunchDimensions raise {
  @launch_contract.AotLaunchDimensions::new(
    grid_x=1,
    grid_y=1,
    grid_z=1,
    block_x=32,
    block_y=1,
    block_z=1,
    shared_memory_bytes=0,
  )
}

///|
fn tp_execution_launch_inputs(
  model : @plan.ModelPlan,
  rank_plan : @tensor_parallel_plan.TensorParallelRankPlan,
  target : @catalog.DeviceTarget,
  digest : @catalog.AotArtifactDigest,
  rank : Int,
) -> Array[
  @tensor_parallel_launch_contract.TensorParallelAotLaunchContractInput,
] raise {
  let world_size = 2
  let profile = tp_execution_profile()
  let output : Array[
    @tensor_parallel_launch_contract.TensorParallelAotLaunchContractInput,
  ] = []
  for operation in model.operations() {
    guard rank_plan.operation(operation.id()) is Some(geometry) else {
      fail("rank-local operation geometry is missing")
    }
    output.push(
      @tensor_parallel_launch_contract.TensorParallelAotLaunchContractInput::new(
        profile_id=profile.id(),
        operation_id=operation.id(),
        entry_point=@catalog.AotKernelEntryPoint::new(
          family=tp_execution_family(operation, digest),
          entry_point_id=@catalog.AotKernelEntryPointId::from_stable_int(
            operation.id().as_int() + 1,
          ),
        ),
        dimensions=tp_execution_dimensions(),
        rank~,
        world_size~,
        device_target=target,
        max_query_rows=profile.max_query_rows(),
        max_query_tokens=profile.max_query_tokens(),
        max_page_table_entries=profile.max_page_table_entries(),
        input_width=geometry.input_width(),
        output_width=geometry.output_width(),
        inner_width=geometry.inner_width(),
        query_head_count=geometry.query_head_count(),
        kv_head_count=geometry.kv_head_count(),
        head_dimension=geometry.head_dimension(),
        kv_layout_version=1,
        tokens_per_page=128,
      ),
    )
  }
  output
}

///|
fn tp_execution_artifacts(
  launch_set : @tensor_parallel_launch_contract.TensorParallelAotLaunchContractSet,
  module_bytes : Bytes,
  digest : @catalog.AotArtifactDigest,
) -> @artifact.KernelArtifactBundle raise {
  let inputs : Array[@artifact.KernelEntryPointInput] = []
  let seen : Array[@catalog.AotKernelEntryPoint] = []
  for contract in launch_set.contracts() {
    if !seen.contains(contract.entry_point()) {
      seen.push(contract.entry_point())
      inputs.push(
        @artifact.KernelEntryPointInput::new(
          entry_point=contract.entry_point(),
          function_symbol="tp_bootstrap_op_\{contract.operation_id().as_int()}",
        ),
      )
    }
  }
  @artifact.admit_tensor_parallel(
    launch_set,
    [@artifact.KernelModuleInput::from_bytes(digest~, module_bytes~)],
    inputs,
    @artifact.KernelArtifactLimits::new(
      max_modules=4,
      max_entry_points=64,
      max_module_bytes=4096L,
      max_total_module_bytes=4096L,
      max_function_symbol_bytes=64,
    ),
  )
}

///|
fn tp_execution_limits() -> @tensor_parallel_execution_plan.TensorParallelExecutionPlanLimits raise {
  @tensor_parallel_execution_plan.TensorParallelExecutionPlanLimits::new(
    max_operations=64,
    max_operands=1024,
    max_descriptors=16,
    max_collectives=32,
    max_descriptor_bytes=1048576L,
    max_activation_workspace_bytes=1048576L,
    artifact_limits=@artifact.KernelArtifactLimits::new(
      max_modules=4,
      max_entry_points=64,
      max_module_bytes=4096L,
      max_total_module_bytes=4096L,
      max_function_symbol_bytes=64,
    ),
  )
}

///|
fn tp_bootstrap_execution_plans(
  model : @plan.ModelPlan,
  inspections : ArrayView[@device_materialize.DeviceShardedWeightFileInspection],
  device_plans : ArrayView[
    @tensor_parallel_device_plan.TensorParallelDevicePlan,
  ],
  kv_plan : @tensor_parallel_kv_plan.TensorParallelKvPlan,
  module_text? : String = "tp-bootstrap-module-primary",
) -> Array[@tensor_parallel_execution_plan.TensorParallelExecutionPlan] raise {
  let module_bytes = @utf8.encode(module_text)
  let digest = @catalog.AotArtifactDigest::from_sha256(
    @crypto.bytes_to_hex_string(@crypto.sha256(module_bytes)),
  )
  let target = device_plans[0].target()
  let catalog = tp_execution_catalog(model, target, digest)
  let output : Array[
    @tensor_parallel_execution_plan.TensorParallelExecutionPlan,
  ] = []
  for rank in 0..<2 {
    let launch_set = @tensor_parallel_launch_contract.admit_paged_rank(
      model,
      inspections[rank].rank_plan(),
      kv_plan.rank_layout(rank),
      catalog,
      expected_rank=rank,
      expected_world_size=2,
      expected_target=target,
      [tp_execution_profile()],
      tp_execution_launch_inputs(
        model,
        inspections[rank].rank_plan(),
        target,
        digest,
        rank,
      ),
      @tensor_parallel_launch_contract.TensorParallelLaunchContractLimits::new(
        max_profiles=2,
        max_contracts=64,
        max_operands_per_contract=16,
      ),
    )
    let artifacts = tp_execution_artifacts(launch_set, module_bytes, digest)
    output.push(
      @tensor_parallel_execution_plan.admit(
        model,
        launch_set,
        device_plans[rank],
        artifacts,
        tp_execution_profile(),
        tp_execution_limits(),
      ),
    )
  }
  output
}
