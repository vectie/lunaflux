///|
priv struct GroupTransportFixture {
  inputs : TensorParallelGroupPlanningInputs
  model_root : @approved_fs.ApprovedRoot
  kernel_root : @approved_fs.ApprovedRoot
  plan_identities : ReadOnlyArray[@worker_protocol.PlanBufferIdentity]
}

///|
fn group_fixture_weight_limits() -> @device_materialize.DeviceWeightFileLimits raise {
  @device_materialize.DeviceWeightFileLimits::new(
    max_file_bytes=16384L,
    transfer_chunk_bytes=17,
    safetensors=@safetensors.HeaderLimits::new(
      max_header_bytes=8192,
      max_tensor_count=32,
      max_rank=2,
      max_name_length=128,
      max_payload_bytes=4096L,
      max_materialized_tensor_bytes=4096L,
    ),
  )
}

///|
fn group_fixture_ordinary_source(
  model_path : String,
  kernel_path : String,
) -> @worker_wire.EncodedBootstrapSource raise {
  @worker_wire.admit_bootstrap_source(
    @worker_wire.WorkerBootstrapRoots::new(
      model_root=model_path,
      kernel_root=kernel_path,
    ),
    @worker_wire.WorkerBootstrapModelSource::new(
      config_locator="config.json",
      weights_locator="weights.safetensors",
      config_digest_sha256="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    ),
    @worker_wire.WorkerBootstrapKernelSource::new(
      manifest_locator="execution.json",
      manifest_digest_sha256="bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
    ),
    @worker_wire.WorkerBootstrapExecution::new(
      compute_major=8,
      compute_minor=0,
      supports_bf16=true,
      supports_cublas_lt=false,
      tokens_per_page=64,
      total_page_count=8,
    ),
    @worker_wire.WorkerBootstrapCeilings::new(
      max_model_config_bytes=4096,
      max_weight_file_bytes=16384L,
      max_weight_arena_bytes=4096L,
      max_activation_arena_bytes=65536L,
      max_kv_arena_bytes=1048576L,
      max_execution_manifest_bytes=65536L,
      max_module_bytes=1024L,
      max_total_module_bytes=1024L,
    ),
  )
}

///|
fn group_fixture_source(
  model_path : String,
  kernel_path : String,
) -> @worker_wire.EncodedBootstrapSource raise {
  let ordinary = group_fixture_ordinary_source(model_path, kernel_path)
  @worker_wire.admit_tensor_parallel_bootstrap_source(
    ordinary.roots(),
    ordinary.model(),
    ordinary.kernel(),
    ordinary.execution(),
    ordinary.ceilings(),
    @worker_wire.TensorParallelBootstrapPolicy::new(
      [
        @worker_wire.TensorParallelRankRuntimeClaim::new(
          rank=0,
          device_ordinal=2,
          memory_ceiling_bytes=1073741824L,
          weight_reserved_bytes=4096L,
          activation_workspace_reserved_bytes=65536L,
          kv_reserved_bytes=1048576L,
          collective_reserved_bytes=1048576L,
        ),
        @worker_wire.TensorParallelRankRuntimeClaim::new(
          rank=1,
          device_ordinal=5,
          memory_ceiling_bytes=1073741824L,
          weight_reserved_bytes=4096L,
          activation_workspace_reserved_bytes=65536L,
          kv_reserved_bytes=1048576L,
          collective_reserved_bytes=1048576L,
        ),
      ],
      minimum_collective_runtime_version_code=21403,
      maximum_collective_runtime_version_code=21403,
      collective_exchange_timeout_millis=600000,
    ),
  )
}

///|
fn group_fixture_approved_source(
  source : @worker_wire.EncodedBootstrapSource,
) -> @worker_wire.EncodedBootstrapSource raise {
  let kernel = source.kernel()
  @worker_wire.admit_tensor_parallel_approved_bootstrap_source(
    source.roots(),
    source.model(),
    @worker_wire.WorkerBootstrapKernelSource::new(
      manifest_locator=kernel.manifest_locator(),
      manifest_digest_sha256=kernel.manifest_digest(),
      luna_approval=Some(
        @worker_wire.WorkerBootstrapLunaApproval::from_untrusted_wire_claim(
          manifest_digest_sha256=kernel.manifest_digest(),
          key_identity_sha256="dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd",
          signature_digest_sha256="eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee",
          approved_source_sha256=kernel.manifest_digest(),
        ),
      ),
    ),
    source.execution(),
    source.ceilings(),
    source.tensor_parallel_policy().unwrap(),
  )
}

///|
fn group_fixture_worker_limits() -> @worker_protocol.WorkerProtocolLimits raise {
  @worker_protocol.WorkerProtocolLimits::new(
    max_prefill_rows=2,
    max_decode_rows=2,
    max_plan_rows=2,
    max_plan_tokens=4,
    max_plan_pages=2,
    max_capabilities=64,
    max_completion_slots=2,
    max_sequence_tokens=16,
    max_token_id=7,
  )
}

///|
fn group_fixture_inference_limits() -> @inference.InferenceLimits raise {
  @inference.InferenceLimits::new(
    max_text_bytes=1024,
    max_input_tokens=16,
    max_new_tokens=16,
    max_context_tokens=16,
    max_token_id=7,
    max_stop_token_ids=8,
    max_stop_strings=4,
    max_stop_string_bytes=64,
    max_trace_bytes=64,
    max_cache_scope_bytes=64,
    max_decoded_delta_bytes=64,
    max_deadline_millis=1000UL,
    max_top_k=8,
    max_temperature=4.0,
  )
}

///|
fn group_fixture_inputs(
  model_path : String,
  kernel_path : String,
  model_root : @approved_fs.ApprovedRoot,
) -> TensorParallelGroupPlanningInputs raise {
  let weights = group_fixture_weight_file()
  let metadata = group_fixture_metadata(weights)
  let model = @llama.build_paged(
    metadata,
    batch=@llama.LlamaPagedBatchEnvelope::new(2),
  )
  let target = group_fixture_target()
  let (declaration, topology) = group_fixture_topology(target)
  let inspections : Array[@device_materialize.DeviceShardedWeightFileInspection] = []
  for rank in 0..<2 {
    inspections.push(
      @device_materialize.inspect_sharded_file(
        model_root,
        metadata,
        model,
        @approved_fs.ApprovedRelativeLocator::new("weights.safetensors"),
        rank~,
        world_size=2,
        alignment_bytes=64L,
        max_arena_bytes=4096L,
        limits=group_fixture_weight_limits(),
      ),
    )
  }
  let artifact_limits = @artifact.KernelArtifactLimits::new(
    max_modules=1,
    max_entry_points=model.operations().length(),
    max_module_bytes=1024L,
    max_total_module_bytes=1024L,
    max_function_symbol_bytes=64,
  )
  let manifest_limits = @execution_manifest_file.ExecutionManifestFileLimits::new(
    max_manifest_bytes=65536L,
    max_json_depth=16,
    max_modules=1,
    max_operations=model.operations().length(),
    device_memory=@device_memory.DeviceMemoryLimits::new(
      activation_alignment=64L,
      max_arena_bytes=65536L,
    ),
    launch_contract=@launch_contract.LaunchContractLimits::new(
      max_profiles=1,
      max_contracts=model.operations().length(),
      max_operands_per_contract=16,
    ),
    artifact_file=@artifact_file.KernelArtifactFileLimits::new(
      max_manifest_bytes=65536L,
      max_json_depth=16,
      max_path_bytes=64,
      artifact=artifact_limits,
    ),
  )
  let worker_limits = group_fixture_worker_limits()
  TensorParallelGroupPlanningInputs::new(
    source=group_fixture_source(model_path, kernel_path),
    model_plan=model,
    model_generation=@worker_protocol.ModelPlanGeneration::from_wire(7UL),
    topology_declaration=declaration,
    topology~,
    inspection_blueprints=inspections,
    device_step_limits=@device_step.DeviceStepLimits::new(
      max_rows=2,
      max_tokens=4,
      max_pages=2,
      max_sequence_tokens=16,
      max_token_id=7,
    ),
    manifest_limits~,
    launch_limits=@tensor_parallel_launch_contract.TensorParallelLaunchContractLimits::new(
      max_profiles=1,
      max_contracts=model.operations().length(),
      max_operands_per_contract=16,
    ),
    execution_limits=@tensor_parallel_execution_plan.TensorParallelExecutionPlanLimits::new(
      max_operations=model.operations().length(),
      max_operands=model.operations().length() * 16,
      max_descriptors=16,
      max_collectives=model.operations().length(),
      max_descriptor_bytes=65536L,
      max_activation_workspace_bytes=65536L,
      artifact_limits~,
    ),
    worker_limits~,
    inference_limits=group_fixture_inference_limits(),
    process_limits=@rank_group_process.RankGroupProcessLimits::new(
      max_payload_bytes=16384,
      startup_io_timeout_millis=1000,
      io_timeout_millis=1000,
      shutdown_timeout_millis=1000,
      collective_exchange_timeout_millis=600000,
    ),
    admitted_rank_child_activation_path=@worker_executable_fixture.cat_path(),
    rank_child_executable_digest_sha256=@worker_executable_fixture.digest(
      @worker_executable_fixture.cat_path(),
    ),
  )
}

///|
async fn group_fixture(directory : String) -> GroupTransportFixture {
  let model_path = "\{directory}/model"
  let kernel_path = "\{directory}/kernel"
  @fs.mkdir(model_path)
  @fs.mkdir(kernel_path)
  @fs.write_file(
    "\{model_path}/weights.safetensors",
    group_fixture_weight_file(),
    create_mode=CreateNew,
  )
  let canonical_model_path = @fs.realpath(model_path)
  let canonical_kernel_path = @fs.realpath(kernel_path)
  let model_root = @approved_fs.ApprovedRoot::open_absolute(
    canonical_model_path,
  )
  let kernel_root = @approved_fs.ApprovedRoot::open_absolute(
    canonical_kernel_path,
  )
  let worker_limits = group_fixture_worker_limits()
  let plan_a = @worker_protocol.SchedulePlanBuffer::new(worker_limits)
  let plan_b = @worker_protocol.SchedulePlanBuffer::new(worker_limits)
  {
    inputs: group_fixture_inputs(
      canonical_model_path, canonical_kernel_path, model_root,
    ),
    model_root,
    kernel_root,
    plan_identities: ReadOnlyArray::from_array([
      plan_a.identity(),
      plan_b.identity(),
    ]),
  }
}
