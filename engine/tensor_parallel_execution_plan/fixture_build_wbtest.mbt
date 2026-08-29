///|
fn exec_local_layout(
  model : @plan.ModelPlan,
) -> @device_layout.DeviceKvLayout raise {
  let full = model.kv_cache_geometry()
  @device_layout.plan(
    model.identity(),
    @plan.KvCacheGeometry::new(
      layer_count=full.layer_count(),
      kv_head_count=1,
      head_dimension=full.head_dimension(),
      dtype=full.dtype(),
      layout=full.layout(),
    ),
    @device_layout.DeviceKvLimits::new(
      tokens_per_page=64,
      total_page_count=8,
      max_arena_bytes=1048576L,
    ),
  )
}

///|
fn exec_full_layout(
  model : @plan.ModelPlan,
) -> @device_layout.DeviceKvLayout raise {
  @device_layout.plan(
    model.identity(),
    model.kv_cache_geometry(),
    @device_layout.DeviceKvLimits::new(
      tokens_per_page=64,
      total_page_count=8,
      max_arena_bytes=1048576L,
    ),
  )
}

///|
fn exec_family(
  operation : @plan.PlanOperation,
  digest : @catalog.AotArtifactDigest,
) -> @catalog.AotKernelFamily raise {
  @catalog.AotKernelFamily::new(
    module_digest=digest,
    family_id=@catalog.AotKernelFamilyId::from_stable_int(
      exec_family_index(operation) + 1,
    ),
  )
}

///|
fn exec_resolved_catalog(
  model : @plan.ModelPlan,
  target : @catalog.DeviceTarget,
  full_layout : @device_layout.DeviceKvLayout,
  digest : @catalog.AotArtifactDigest,
) -> @catalog.ResolvedKernelCatalog raise {
  let entries : Array[@catalog.CatalogEntry] = []
  for index, operation in model.operations() {
    let family_index = exec_family_index(operation)
    if index != family_index {
      continue
    }
    entries.push(
      @catalog.CatalogEntry::new_paged_v3(
        operation_kind=operation.kind(),
        operation_shape=operation.shape(),
        capability=operation.required_capability(),
        execution_semantics=exec_semantics(operation),
        compute_major=target.compute_major(),
        compute_minor=target.compute_minor(),
        implementation=ContentAddressedAot(exec_family(operation, digest)),
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
    model, full_layout, target,
  )
}

///|
fn exec_launch_inputs(
  model : @plan.ModelPlan,
  rank_plan : @tensor_parallel_plan.TensorParallelRankPlan,
  profiles : ArrayView[@launch_contract.PagedKernelProfile],
  target : @catalog.DeviceTarget,
  digest : @catalog.AotArtifactDigest,
  rank : Int,
) -> Array[
  @tensor_parallel_launch_contract.TensorParallelAotLaunchContractInput,
] raise {
  let output : Array[
    @tensor_parallel_launch_contract.TensorParallelAotLaunchContractInput,
  ] = []
  for profile in profiles {
    for index, operation in model.operations() {
      let geometry = rank_plan.operation(operation.id()).unwrap()
      output.push(
        @tensor_parallel_launch_contract.TensorParallelAotLaunchContractInput::new(
          profile_id=profile.id(),
          operation_id=operation.id(),
          entry_point=@catalog.AotKernelEntryPoint::new(
            family=exec_family(operation, digest),
            entry_point_id=@catalog.AotKernelEntryPointId::from_stable_int(
              index + 1,
            ),
          ),
          dimensions=exec_dimensions(),
          rank~,
          world_size=2,
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
          tokens_per_page=64,
        ),
      )
    }
  }
  output
}

///|
fn exec_artifacts(
  launch_set : @tensor_parallel_launch_contract.TensorParallelAotLaunchContractSet,
  module_bytes : Bytes,
  digest : @catalog.AotArtifactDigest,
) -> @artifact.KernelArtifactBundle raise {
  let entry_inputs : Array[@artifact.KernelEntryPointInput] = []
  let seen : Array[@catalog.AotKernelEntryPoint] = []
  for contract in launch_set.contracts() {
    if seen.contains(contract.entry_point()) {
      continue
    }
    seen.push(contract.entry_point())
    entry_inputs.push(
      @artifact.KernelEntryPointInput::new(
        entry_point=contract.entry_point(),
        function_symbol="tp_op_\{contract.operation_id().as_int()}",
      ),
    )
  }
  @artifact.admit_tensor_parallel(
    launch_set,
    [@artifact.KernelModuleInput::from_bytes(digest~, module_bytes~)],
    entry_inputs,
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
fn build_execution_fixture(
  root : @approved_fs.ApprovedRoot,
  file : BytesView,
  rank : Int,
  module_text : String,
) -> ExecutionFixture raise {
  let metadata = exec_metadata(file)
  let model = @llama.build_paged(
    metadata,
    batch=@llama.LlamaPagedBatchEnvelope::new(2),
  )
  let inspection = @device_materialize.inspect_sharded_file(
    root,
    metadata,
    model,
    @approved_fs.ApprovedRelativeLocator::new("weights.safetensors"),
    rank~,
    world_size=2,
    alignment_bytes=64L,
    max_arena_bytes=16384L,
    limits=exec_file_limits(),
  )
  let target = exec_target()
  let device_plan = @tensor_parallel_device_plan.build(
    model,
    inspection.rank_plan(),
    inspection,
    @tensor_parallel_device_plan.TensorParallelRankDevice::from_local_topology(
      exec_topology(target),
      rank~,
    ),
    target,
  )
  let module_bytes = @utf8.encode(module_text)
  let digest = @catalog.AotArtifactDigest::from_sha256(
    @crypto.bytes_to_hex_string(@crypto.sha256(module_bytes)),
  )
  let full_layout = exec_full_layout(model)
  let local_layout = exec_local_layout(model)
  let resolved = exec_resolved_catalog(model, target, full_layout, digest)
  let profiles = [exec_profile(1, 4), exec_profile(2, 4)]
  let launch_set = @tensor_parallel_launch_contract.admit_paged_rank(
    model,
    inspection.rank_plan(),
    local_layout,
    resolved,
    expected_rank=rank,
    expected_world_size=2,
    expected_target=target,
    profiles,
    exec_launch_inputs(
      model,
      inspection.rank_plan(),
      profiles,
      target,
      digest,
      rank,
    ),
    @tensor_parallel_launch_contract.TensorParallelLaunchContractLimits::new(
      max_profiles=4,
      max_contracts=64,
      max_operands_per_contract=16,
    ),
  )
  {
    model,
    launch_set,
    device_plan,
    artifacts: exec_artifacts(launch_set, module_bytes, digest),
    profiles,
  }
}
