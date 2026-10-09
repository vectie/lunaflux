///|
fn compiled_plan() -> @two_node_plan.TwoNodePlan raise {
  let (model, bindings) = tp_fixture()
  let target = @catalog.DeviceTarget::new(
    compute_major=12,
    compute_minor=1,
    supports_bf16=true,
    supports_cublas_lt=true,
  )
  let nodes = [
    for label in ["spark-a", "spark-b"] => {
      @network_topology.NetworkRank::new(
        node=@network_topology.NodeIdentity::new(label),
        device_ordinal=0,
        target~,
        physical_memory_bytes=128000000000L,
        rdma_interface="mlx5_0",
      )
    }
  ]
  let budget = @two_node_plan.RankMemoryBudget::new(
    system_reserved_bytes=8000000000L,
    runtime_ceiling_bytes=120000000000L,
    activation_workspace_bytes=1024L,
    collective_bytes=1024L,
    kv_bytes=1048576L,
  )
  @two_node_plan.compile(
    model,
    @network_topology.TwoNodeTopology::new(nodes[0], nodes[1]),
    [tp_plan(model, bindings, 0, 2), tp_plan(model, bindings, 1, 2)],
    [budget, budget],
    tokens_per_page=4,
    maximum_pages=1024,
  )
}

///|
fn digest(character : Char) -> @spec.ContentDigest raise {
  @spec.ContentDigest::from_sha256(String::make(64, character))
}

///|
fn worker(
  plan : @two_node_plan.TwoNodePlan,
  rank : Int,
  generation? : UInt64 = 1UL,
  predecessor? : UInt64 = 0UL,
  ordinal? : Int = 0,
  input_tokens? : Int = 64,
) -> @worker_wire.WorkerStartupContract raise {
  @worker_wire.WorkerStartupContract::new(
    identity=plan.model_identity(),
    bootstrap_digest=@worker_wire.WorkerBootstrapDigest::from_sha256(
      digest(if rank == 0 { 'a' } else { 'b' }).as_hex(),
    ),
    bootstrap_source_digest=@worker_wire.WorkerBootstrapSourceDigest::from_sha256(
      digest(if rank == 0 { 'c' } else { 'd' }).as_hex(),
    ),
    device_ordinal=ordinal,
    model_generation_value=generation,
    worker_limits=@worker_protocol.WorkerProtocolLimits::new(
      max_prefill_rows=4,
      max_decode_rows=4,
      max_plan_rows=8,
      max_plan_tokens=32,
      max_plan_pages=32,
      max_capabilities=32,
      max_completion_slots=8,
      max_sequence_tokens=64,
      max_token_id=1024,
    ),
    inference_limits=@inference.InferenceLimits::new(
      max_text_bytes=1024,
      max_input_tokens=input_tokens,
      max_new_tokens=64,
      max_context_tokens=64,
      max_token_id=1024,
      max_stop_token_ids=8,
      max_stop_strings=1,
      max_stop_string_bytes=1,
      max_trace_bytes=64,
      max_cache_scope_bytes=64,
      max_decoded_delta_bytes=64,
      max_deadline_millis=1000UL,
      max_top_k=32,
      max_temperature=4.0,
    ),
    has_predecessor=predecessor != 0UL,
    predecessor_value=predecessor,
  )
}

///|
fn rendezvous(value? : Byte = b'\x01') -> @device.TensorCollectiveGroupId raise {
  @device.tensor_collective_group_id_from_bytes(Bytes::make(128, value))
}

///|
fn group(
  plan : @two_node_plan.TwoNodePlan,
  generation? : UInt64 = 1UL,
  nonce? : Byte = b'\x01',
  lease? : UInt64 = 1000UL,
  follower_executable? : Char = 'f',
  predecessor? : UInt64 = 0UL,
) -> @remote_startup.RemoteGroupStartup raise {
  @remote_startup.RemoteGroupStartup::new(
    plan,
    [worker(plan, 0, predecessor~), worker(plan, 1, predecessor~)],
    [digest('e'), digest(follower_executable)],
    deployment_digest=digest('0'),
    group_generation=generation,
    rendezvous=rendezvous(value=nonce),
    lease_millis=lease,
  )
}
