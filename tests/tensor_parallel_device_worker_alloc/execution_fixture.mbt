///|
priv struct RankExecutionFixture {
  launch_set : @tensor_parallel_launch_contract.TensorParallelAotLaunchContractSet
  artifacts : @artifact.KernelArtifactBundle
  execution : @tensor_parallel_execution_plan.TensorParallelExecutionPlan
}

///|
priv struct ExecutionFixture {
  model : @plan.ModelPlan
  topology : TopologyFixture
  inspections : Array[@device_materialize.DeviceShardedWeightFileInspection]
  devices : Array[@tensor_parallel_device_plan.TensorParallelDevicePlan]
  kv_plan : @tensor_parallel_kv_plan.TensorParallelKvPlan
  ranks : Array[RankExecutionFixture]
  profile : @launch_contract.PagedKernelProfile
  execution_limits : @tensor_parallel_execution_plan.TensorParallelExecutionPlanLimits
}

///|
fn family_index(operation : @plan.PlanOperation) -> Int {
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
fn fixture_family(
  operation : @plan.PlanOperation,
  digest : @catalog.AotArtifactDigest,
) -> @catalog.AotKernelFamily raise {
  @catalog.AotKernelFamily::new(
    module_digest=digest,
    family_id=@catalog.AotKernelFamilyId::from_stable_int(
      family_index(operation) + 1,
    ),
  )
}

///|
fn fixture_semantics(
  operation : @plan.PlanOperation,
) -> @catalog.KernelExecutionSemantics raise {
  let kv_layout = match operation.persistent_state() {
    NoPersistentState => None
    KvCacheReadWrite(_) =>
      Some(
        @catalog.PagedKvLayoutContract::new(
          layout=LayerMajorSplitKeyValueV1,
          stable_version=1,
          tokens_per_page=TP_TOKENS_PER_PAGE,
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
fn fixture_full_layout(
  model : @plan.ModelPlan,
) -> @device_layout.DeviceKvLayout raise {
  @device_layout.plan(
    model.identity(),
    model.kv_cache_geometry(),
    @device_layout.DeviceKvLimits::new(
      tokens_per_page=TP_TOKENS_PER_PAGE,
      total_page_count=TP_TOTAL_PAGE_COUNT,
      max_arena_bytes=1048576L,
    ),
  )
}

///|
fn fixture_catalog(
  model : @plan.ModelPlan,
  target : @catalog.DeviceTarget,
  digest : @catalog.AotArtifactDigest,
) -> @catalog.ResolvedKernelCatalog raise {
  let entries : Array[@catalog.CatalogEntry] = []
  for index, operation in model.operations() {
    if index != family_index(operation) {
      continue
    }
    entries.push(
      @catalog.CatalogEntry::new_paged_v3(
        operation_kind=operation.kind(),
        operation_shape=operation.shape(),
        capability=operation.required_capability(),
        execution_semantics=fixture_semantics(operation),
        compute_major=target.compute_major(),
        compute_minor=target.compute_minor(),
        implementation=ContentAddressedAot(fixture_family(operation, digest)),
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
    fixture_full_layout(model),
    target,
  )
}

///|
fn fixture_profile() -> @launch_contract.PagedKernelProfile raise {
  @launch_contract.PagedKernelProfile::new(
    id=@launch_contract.KernelProfileId::from_stable_int(1),
    max_query_rows=TP_MAX_ROWS,
    max_query_tokens=TP_MAX_TOKENS,
    max_page_table_entries=TP_MAX_PAGES,
  )
}

///|
fn fixture_dimensions() -> @launch_contract.AotLaunchDimensions raise {
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
fn launch_inputs(
  model : @plan.ModelPlan,
  rank_plan : @tensor_parallel_plan.TensorParallelRankPlan,
  target : @catalog.DeviceTarget,
  digest : @catalog.AotArtifactDigest,
  rank : Int,
  profile : @launch_contract.PagedKernelProfile,
) -> Array[
  @tensor_parallel_launch_contract.TensorParallelAotLaunchContractInput,
] raise {
  let output : Array[
    @tensor_parallel_launch_contract.TensorParallelAotLaunchContractInput,
  ] = []
  for operation in model.operations() {
    guard rank_plan.operation(operation.id()) is Some(geometry) else {
      abort("missing rank-local operation geometry")
    }
    output.push(
      @tensor_parallel_launch_contract.TensorParallelAotLaunchContractInput::new(
        profile_id=profile.id(),
        operation_id=operation.id(),
        entry_point=@catalog.AotKernelEntryPoint::new(
          family=fixture_family(operation, digest),
          entry_point_id=@catalog.AotKernelEntryPointId::from_stable_int(
            operation.id().as_int() + 1,
          ),
        ),
        dimensions=fixture_dimensions(),
        rank~,
        world_size=TP_WORLD_SIZE,
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
        tokens_per_page=TP_TOKENS_PER_PAGE,
      ),
    )
  }
  output
}

///|
fn artifact_limits() -> @artifact.KernelArtifactLimits raise {
  @artifact.KernelArtifactLimits::new(
    max_modules=1,
    max_entry_points=64,
    max_module_bytes=4096L,
    max_total_module_bytes=4096L,
    max_function_symbol_bytes=64,
  )
}

///|
fn execution_limits() -> @tensor_parallel_execution_plan.TensorParallelExecutionPlanLimits raise {
  @tensor_parallel_execution_plan.TensorParallelExecutionPlanLimits::new(
    max_operations=64,
    max_operands=1024,
    max_descriptors=16,
    max_collectives=32,
    max_descriptor_bytes=1048576L,
    max_activation_workspace_bytes=1048576L,
    artifact_limits=artifact_limits(),
  )
}

///|
fn fixture_artifacts(
  launch_set : @tensor_parallel_launch_contract.TensorParallelAotLaunchContractSet,
  module_bytes : Bytes,
  digest : @catalog.AotArtifactDigest,
) -> @artifact.KernelArtifactBundle raise {
  let inputs : Array[@artifact.KernelEntryPointInput] = []
  for contract in launch_set.contracts() {
    inputs.push(
      @artifact.KernelEntryPointInput::new(
        entry_point=contract.entry_point(),
        function_symbol="tp_alloc_op_\{contract.operation_id().as_int()}",
      ),
    )
  }
  @artifact.admit_tensor_parallel(
    launch_set,
    [@artifact.KernelModuleInput::from_bytes(digest~, module_bytes~)],
    inputs,
    artifact_limits(),
  )
}

///|
fn prepare_execution_fixture(
  root : @approved_fs.ApprovedRoot,
  weights : BytesView,
) -> ExecutionFixture raise {
  let metadata = fixture_metadata(weights)
  let model = @llama.build_paged(
    metadata,
    batch=@llama.LlamaPagedBatchEnvelope::new(TP_MAX_ROWS),
  )
  let target = fixture_target()
  let topology = prepare_topology_fixture(target)
  let inspections : Array[@device_materialize.DeviceShardedWeightFileInspection] = []
  let devices : Array[@tensor_parallel_device_plan.TensorParallelDevicePlan] = []
  for rank in 0..<TP_WORLD_SIZE {
    let inspection = @device_materialize.inspect_sharded_file(
      root,
      metadata,
      model,
      @approved_fs.ApprovedRelativeLocator::new("weights.safetensors"),
      rank~,
      world_size=TP_WORLD_SIZE,
      alignment_bytes=64L,
      max_arena_bytes=4096L,
      limits=fixture_weight_limits(),
    )
    inspections.push(inspection)
    devices.push(
      @tensor_parallel_device_plan.build(
        model,
        inspection.rank_plan(),
        inspection,
        @tensor_parallel_device_plan.TensorParallelRankDevice::from_local_topology(
          topology.topology,
          rank~,
        ),
        target,
      ),
    )
  }
  let runtime = @tensor_parallel_worker_bootstrap.admit_runtime_policy(
    topology.source,
    topology.topology,
  )
  let model_generation = @worker_protocol.ModelPlanGeneration::from_wire(
    TP_MODEL_GENERATION,
  )
  let kv_plan = @tensor_parallel_kv_plan.admit_from_topology(
    model,
    model_generation,
    topology.topology,
    runtime,
    tokens_per_page=TP_TOKENS_PER_PAGE,
    total_page_count=TP_TOTAL_PAGE_COUNT,
  )
  let module_bytes = b"tp-allocation-evidence-module"
  let digest = @catalog.AotArtifactDigest::from_sha256(
    @crypto.bytes_to_hex_string(@crypto.sha256(module_bytes)),
  )
  let catalog = fixture_catalog(model, target, digest)
  let profile = fixture_profile()
  let limits = execution_limits()
  let ranks : Array[RankExecutionFixture] = []
  for rank in 0..<TP_WORLD_SIZE {
    let launch_set = @tensor_parallel_launch_contract.admit_paged_rank(
      model,
      inspections[rank].rank_plan(),
      kv_plan.rank_layout(rank),
      catalog,
      expected_rank=rank,
      expected_world_size=TP_WORLD_SIZE,
      expected_target=target,
      [profile],
      launch_inputs(
        model,
        inspections[rank].rank_plan(),
        target,
        digest,
        rank,
        profile,
      ),
      @tensor_parallel_launch_contract.TensorParallelLaunchContractLimits::new(
        max_profiles=1,
        max_contracts=64,
        max_operands_per_contract=16,
      ),
    )
    let artifacts = fixture_artifacts(launch_set, module_bytes, digest)
    let execution = @tensor_parallel_execution_plan.admit(
      model,
      launch_set,
      devices[rank],
      artifacts,
      profile,
      limits,
    )
    ranks.push({ launch_set, artifacts, execution })
  }
  {
    model,
    topology,
    inspections,
    devices,
    kv_plan,
    ranks,
    profile,
    execution_limits: limits,
  }
}
