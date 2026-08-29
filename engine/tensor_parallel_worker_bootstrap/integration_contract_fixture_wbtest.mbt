///|
fn tp_bootstrap_worker_limits() -> @worker_protocol.WorkerProtocolLimits {
  tp_bootstrap_worker_limits_with_plan_tokens(32)
}

///|
fn tp_bootstrap_worker_limits_with_plan_tokens(
  max_plan_tokens : Int,
) -> @worker_protocol.WorkerProtocolLimits {
  try! @worker_protocol.WorkerProtocolLimits::new(
    max_prefill_rows=4,
    max_decode_rows=4,
    max_plan_rows=8,
    max_plan_tokens~,
    max_plan_pages=32,
    max_capabilities=32,
    max_completion_slots=8,
    max_sequence_tokens=256,
    max_token_id=1024,
  )
}

///|
fn tp_bootstrap_inference_limits() -> @inference.InferenceLimits {
  try! @inference.InferenceLimits::new(
    max_text_bytes=1024,
    max_input_tokens=64,
    max_new_tokens=64,
    max_context_tokens=128,
    max_token_id=1024,
    max_stop_token_ids=8,
    max_stop_strings=4,
    max_stop_string_bytes=64,
    max_trace_bytes=64,
    max_cache_scope_bytes=64,
    max_decoded_delta_bytes=64,
    max_deadline_millis=1000UL,
    max_top_k=32,
    max_temperature=4.0,
  )
}

///|
fn tp_bootstrap_source(
  maximum_collective_runtime_version_code? : Int = 21403,
) -> @worker_wire.EncodedBootstrapSource raise {
  let ordinary = @worker_wire.admit_bootstrap_source(
    @worker_wire.WorkerBootstrapRoots::new(
      model_root="/srv/lunaflux/models",
      kernel_root="/srv/lunaflux/kernels",
    ),
    @worker_wire.WorkerBootstrapModelSource::new(
      config_locator="config.json",
      weights_locator="weights.safetensors",
      config_digest_sha256="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    ),
    @worker_wire.WorkerBootstrapKernelSource::new(
      manifest_locator="tensor-parallel.manifest",
      manifest_digest_sha256="bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
    ),
    @worker_wire.WorkerBootstrapExecution::new(
      compute_major=8,
      compute_minor=0,
      supports_bf16=true,
      supports_cublas_lt=true,
      tokens_per_page=128,
      total_page_count=4,
    ),
    @worker_wire.WorkerBootstrapCeilings::new(
      max_model_config_bytes=1048576,
      max_weight_file_bytes=16384L,
      max_weight_arena_bytes=4096L,
      max_activation_arena_bytes=1048576L,
      max_kv_arena_bytes=1048576L,
      max_execution_manifest_bytes=1048576L,
      max_module_bytes=1048576L,
      max_total_module_bytes=2097152L,
    ),
  )
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
          activation_workspace_reserved_bytes=0L,
          kv_reserved_bytes=1048576L,
          collective_reserved_bytes=1048576L,
        ),
        @worker_wire.TensorParallelRankRuntimeClaim::new(
          rank=1,
          device_ordinal=5,
          memory_ceiling_bytes=1073741824L,
          weight_reserved_bytes=4096L,
          activation_workspace_reserved_bytes=0L,
          kv_reserved_bytes=1048576L,
          collective_reserved_bytes=1048576L,
        ),
      ],
      minimum_collective_runtime_version_code=21403,
      maximum_collective_runtime_version_code~,
      collective_exchange_timeout_millis=600000,
    ),
  )
}

///|
fn tp_bootstrap_startup(
  identity : @spec.ModelIdentity,
  rank : Int,
  source? : @worker_wire.EncodedBootstrapSource = try! tp_bootstrap_source(),
  worker_limits? : @worker_protocol.WorkerProtocolLimits = tp_bootstrap_worker_limits(),
  has_predecessor? : Bool = false,
  predecessor_value? : UInt64 = 0UL,
) -> @worker_wire.WorkerStartupContract {
  try! @worker_wire.WorkerStartupContract::new(
    identity~,
    bootstrap_digest=@worker_wire.WorkerBootstrapDigest::from_sha256(
      if rank == 0 {
        "1111111111111111111111111111111111111111111111111111111111111111"
      } else {
        "2222222222222222222222222222222222222222222222222222222222222222"
      },
    ),
    bootstrap_source_digest=source.digest(),
    device_ordinal=if rank == 0 { 2 } else { 5 },
    model_generation_value=7UL,
    worker_limits~,
    inference_limits=tp_bootstrap_inference_limits(),
    has_predecessor~,
    predecessor_value~,
  )
}

///|
fn tp_bootstrap_inspections(
  root : @approved_fs.ApprovedRoot,
  model : @spec.LlamaModelMetadata,
  model_plan : @plan.ModelPlan,
) -> Array[@device_materialize.DeviceShardedWeightFileInspection] raise {
  let output : Array[@device_materialize.DeviceShardedWeightFileInspection] = []
  for rank in 0..<2 {
    output.push(
      @device_materialize.inspect_sharded_file(
        root,
        model,
        model_plan,
        @approved_fs.ApprovedRelativeLocator::new("weights.safetensors"),
        rank~,
        world_size=2,
        alignment_bytes=64L,
        max_arena_bytes=4096L,
        limits=tp_bootstrap_file_limits(),
      ),
    )
  }
  output
}

///|
fn tp_bootstrap_device_plans(
  topology : @device_topology.AdmittedLocalTopology,
  model_plan : @plan.ModelPlan,
  inspections : ArrayView[@device_materialize.DeviceShardedWeightFileInspection],
) -> Array[@tensor_parallel_device_plan.TensorParallelDevicePlan] raise {
  let output : Array[@tensor_parallel_device_plan.TensorParallelDevicePlan] = []
  for rank in 0..<2 {
    let inspection = inspections[rank]
    output.push(
      @tensor_parallel_device_plan.build(
        model_plan,
        inspection.rank_plan(),
        inspection,
        @tensor_parallel_device_plan.TensorParallelRankDevice::from_local_topology(
          topology,
          rank~,
        ),
        topology.target(),
      ),
    )
  }
  output
}

///|
fn tp_bootstrap_decode(
  envelope : TensorParallelRankBootstrapEnvelope,
) -> TensorParallelRankBootstrapEnvelope raise TensorParallelWorkerBootstrapError {
  let sender = TensorParallelRankEnvelopeFrameBuffer::new()
  let frame = sender.encode(envelope)
  let wire = FixedArray::make(frame.length(), b'\x00')
  frame.copy_to(wire, destination_offset=0)
  TensorParallelRankEnvelopeFrameBuffer::new()
  .load(wire, wire.length())
  .envelope()
}

///|
fn tp_bootstrap_expect_local_rule(
  body : () -> Unit raise,
  expected : TensorParallelWorkerBootstrapRule,
) -> Unit raise {
  try body() catch {
    Rejected(rule~, rank=_) => assert_eq(rule, expected)
    _ => fail("unexpected local bootstrap rejection")
  } noraise {
    _ => fail("substituted local bootstrap evidence was admitted")
  }
}
