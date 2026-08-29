///|
fn configure_test_identity() -> @spec.ModelIdentity {
  try! @spec.ModelIdentity::new(
    @spec.ContentDigest::from_sha256(
      "1111111111111111111111111111111111111111111111111111111111111111",
    ),
    @spec.PlanDigest::from_sha256(
      "2222222222222222222222222222222222222222222222222222222222222222",
    ),
  )
}

///|
fn configure_test_source() -> @worker_wire.EncodedBootstrapSource raise {
  let ordinary = @worker_wire.admit_bootstrap_source(
    @worker_wire.WorkerBootstrapRoots::new(
      model_root="/srv/lunaflux/models",
      kernel_root="/srv/lunaflux/kernels",
    ),
    @worker_wire.WorkerBootstrapModelSource::new(
      config_locator="config.json",
      weights_locator="weights/model.safetensors",
      config_digest_sha256="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    ),
    @worker_wire.WorkerBootstrapKernelSource::new(
      manifest_locator="manifests/paged-v2.json",
      manifest_digest_sha256="bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
    ),
    @worker_wire.WorkerBootstrapExecution::new(
      compute_major=8,
      compute_minor=0,
      supports_bf16=true,
      supports_cublas_lt=true,
      tokens_per_page=16,
      total_page_count=4096,
    ),
    @worker_wire.WorkerBootstrapCeilings::new(
      max_model_config_bytes=1048576,
      max_weight_file_bytes=8589934592L,
      max_weight_arena_bytes=8589934592L,
      max_activation_arena_bytes=1073741824L,
      max_kv_arena_bytes=4294967296L,
      max_execution_manifest_bytes=1048576L,
      max_module_bytes=67108864L,
      max_total_module_bytes=268435456L,
    ),
  )
  let ranks = [
    @worker_wire.TensorParallelRankRuntimeClaim::new(
      rank=0,
      device_ordinal=4,
      memory_ceiling_bytes=1073741824L,
      weight_reserved_bytes=268435456L,
      activation_workspace_reserved_bytes=134217728L,
      kv_reserved_bytes=268435456L,
      collective_reserved_bytes=0L,
    ),
    @worker_wire.TensorParallelRankRuntimeClaim::new(
      rank=1,
      device_ordinal=5,
      memory_ceiling_bytes=1073741824L,
      weight_reserved_bytes=268435456L,
      activation_workspace_reserved_bytes=134217728L,
      kv_reserved_bytes=268435456L,
      collective_reserved_bytes=67108864L,
    ),
  ]
  @worker_wire.admit_tensor_parallel_bootstrap_source(
    ordinary.roots(),
    ordinary.model(),
    ordinary.kernel(),
    ordinary.execution(),
    ordinary.ceilings(),
    @worker_wire.TensorParallelBootstrapPolicy::new(
      ranks,
      minimum_collective_runtime_version_code=21403,
      maximum_collective_runtime_version_code=21999,
      collective_exchange_timeout_millis=600000,
    ),
  )
}

///|
fn configure_test_worker_limits() -> @worker_protocol.WorkerProtocolLimits {
  try! @worker_protocol.WorkerProtocolLimits::new(
    max_prefill_rows=2,
    max_decode_rows=2,
    max_plan_rows=2,
    max_plan_tokens=8,
    max_plan_pages=8,
    max_capabilities=8,
    max_completion_slots=2,
    max_sequence_tokens=16,
    max_token_id=31,
  )
}

///|
fn configure_test_inference_limits() -> @inference.InferenceLimits {
  try! @inference.InferenceLimits::new(
    max_text_bytes=8,
    max_input_tokens=8,
    max_new_tokens=8,
    max_context_tokens=16,
    max_token_id=31,
    max_stop_token_ids=1,
    max_stop_strings=1,
    max_stop_string_bytes=1,
    max_trace_bytes=1,
    max_cache_scope_bytes=1,
    max_decoded_delta_bytes=1,
    max_deadline_millis=1000UL,
    max_top_k=31,
    max_temperature=4.0,
  )
}

///|
fn configure_test_startup(
  rank : Int,
  source : @worker_wire.EncodedBootstrapSource,
  predecessor? : UInt64 = 6UL,
) -> @worker_wire.WorkerStartupContract {
  try! @worker_wire.WorkerStartupContract::new(
    identity=configure_test_identity(),
    bootstrap_digest=@worker_wire.WorkerBootstrapDigest::from_sha256(
      if rank == 0 {
        "3333333333333333333333333333333333333333333333333333333333333333"
      } else {
        "3434343434343434343434343434343434343434343434343434343434343434"
      },
    ),
    bootstrap_source_digest=source.digest(),
    device_ordinal=4 + rank,
    model_generation_value=7UL,
    worker_limits=configure_test_worker_limits(),
    inference_limits=configure_test_inference_limits(),
    has_predecessor=predecessor != 0UL,
    predecessor_value=predecessor,
  )
}

///|
fn configure_test_topology(
  swapped? : Bool = false,
) -> @device_topology.LocalTopologyDeclaration {
  let target = try! @catalog.DeviceTarget::new(
    compute_major=8,
    compute_minor=0,
    supports_bf16=true,
    supports_cublas_lt=true,
  )
  let first = if swapped { 5 } else { 4 }
  let second = if swapped { 6 } else { 5 }
  try! @device_topology.LocalTopologyDeclaration::new(
    [
      @device_topology.LocalTopologyRank::new(
        ordinal=first,
        name="rank-zero",
        total_memory_bytes=2147483648L,
        target~,
      ),
      @device_topology.LocalTopologyRank::new(
        ordinal=second,
        name="rank-one",
        total_memory_bytes=2147483648L,
        target~,
      ),
    ],
    [@device_topology.LocalTopologyLink::direct(lower_rank=0, upper_rank=1)],
  )
}

///|
fn configure_test_write_digest(
  bytes : FixedArray[Byte],
  offset : Int,
  value : String,
) -> Unit {
  let encoded = @utf8.encode(value)
  for index in 0..<64 {
    bytes[offset + index] = encoded[index]
  }
}

///|
fn configure_test_envelope_checksum(bytes : FixedArray[Byte]) -> UInt {
  let mut hash : UInt = 2166136261
  for index in 0..<752 {
    if index < 12 || index >= 16 {
      hash = (hash ^ bytes[index].to_uint()) * 16777619
    }
  }
  hash
}

///|
fn configure_test_envelope(
  rank : Int,
  startup : @worker_wire.WorkerStartupContract,
  group_generation? : UInt64 = 9UL,
  collective_runtime_version_code? : Int = 21403,
  execution_plan_digest? : String = "9999999999999999999999999999999999999999999999999999999999999999",
) -> @tensor_parallel_worker_bootstrap.TensorParallelRankBootstrapEnvelope {
  let worker_digest = try! @tensor_parallel_worker_bootstrap.worker_contract_digest(
    startup,
  )
  let bytes = FixedArray::make(752, b'\x00')
  configure_write_u32(bytes, 0, 0x4254504c)
  configure_write_u32(bytes, 4, 3)
  configure_write_i32(bytes, 8, 752)
  configure_write_i32(bytes, 16, rank)
  configure_write_i32(bytes, 20, 2)
  configure_write_i32(bytes, 24, 4 + rank)
  configure_write_i32(bytes, 28, collective_runtime_version_code)
  configure_write_u64(bytes, 32, 7UL)
  configure_write_u64(bytes, 40, group_generation)
  configure_test_write_digest(
    bytes,
    48,
    configure_test_identity().content().as_hex(),
  )
  configure_test_write_digest(
    bytes,
    112,
    configure_test_identity().plan().as_hex(),
  )
  configure_test_write_digest(
    bytes, 176, "7777777777777777777777777777777777777777777777777777777777777777",
  )
  configure_test_write_digest(
    bytes, 240, "8888888888888888888888888888888888888888888888888888888888888888",
  )
  configure_test_write_digest(bytes, 304, worker_digest.as_sha256())
  configure_test_write_digest(
    bytes,
    368,
    if rank == 0 {
      "4444444444444444444444444444444444444444444444444444444444444444"
    } else {
      "4545454545454545454545454545454545454545454545454545454545454545"
    },
  )
  configure_test_write_digest(bytes, 432, execution_plan_digest)
  configure_test_write_digest(
    bytes, 496, "5555555555555555555555555555555555555555555555555555555555555555",
  )
  configure_test_write_digest(
    bytes, 560, "6666666666666666666666666666666666666666666666666666666666666666",
  )
  for index in 0..<128 {
    bytes[624 + index] = b'\x5a'
  }
  configure_write_u32(bytes, 12, configure_test_envelope_checksum(bytes))
  try! @tensor_parallel_worker_bootstrap.TensorParallelRankEnvelopeFrameBuffer::new()
  .load(bytes, 752)
  .envelope()
}

///|
test "rank configure rejects re-signed execution-plan substitution" {
  let source = configure_test_source()
  let startup = configure_test_startup(0, source)
  let binding = configure_test_binding(0, startup)
  let substituted = configure_test_envelope(
    0,
    startup,
    execution_plan_digest="eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee",
  )
  try
    compose_rank_configure(
      binding,
      startup,
      source,
      configure_test_topology(),
      substituted,
    )
  catch {
    Invalid(Binding) => ()
    _ => fail("unexpected execution-plan substitution rejection")
  } noraise {
    _ => fail("re-signed inner execution plan escaped outer binding")
  }
}

///|
fn configure_test_binding(
  rank : Int,
  startup : @worker_wire.WorkerStartupContract,
  group_generation? : UInt64 = 9UL,
) -> @rank_group_wire.RankGroupWireBinding {
  let worker_digest = try! @tensor_parallel_worker_bootstrap.worker_contract_digest(
    startup,
  )
  try! @rank_group_wire.RankGroupWireBinding::new(
    identity=configure_test_identity(),
    model_generation_value=7UL,
    group_generation_value=group_generation,
    rank~,
    world_size=2,
    group_digest="7777777777777777777777777777777777777777777777777777777777777777",
    topology_digest="8888888888888888888888888888888888888888888888888888888888888888",
    worker_digest=worker_digest.as_sha256(),
    device_plan_digest=if rank == 0 {
      "4444444444444444444444444444444444444444444444444444444444444444"
    } else {
      "4545454545454545454545454545454545454545454545454545454545454545"
    },
    execution_plan_digest="9999999999999999999999999999999999999999999999999999999999999999",
    collective_digest="5555555555555555555555555555555555555555555555555555555555555555",
    kv_digest="6666666666666666666666666666666666666666666666666666666666666666",
  )
}

///|
fn configure_test_contract(
  rank? : Int = 0,
) -> TensorParallelRankConfigureContract raise {
  let source = configure_test_source()
  let startup = configure_test_startup(rank, source)
  compose_rank_configure(
    configure_test_binding(rank, startup),
    startup,
    source,
    configure_test_topology(),
    configure_test_envelope(rank, startup),
  )
}

///|
test "canonical rank configure roundtrips and stale views revoke" {
  let contract = configure_test_contract()
  assert_true(contract.length() < tensor_parallel_rank_configure_capacity())
  assert_eq(contract.digest().as_sha256().length(), 64)
  assert_eq(contract.startup().predecessor_value(), 6UL)
  assert_eq(
    contract.bootstrap_source().digest(),
    contract.startup().bootstrap_source_digest(),
  )
  assert_eq(contract.topology().rank_count(), 2)
  assert_eq(contract.rank_envelope().rank(), 0)
  let sender = TensorParallelRankConfigureFrameBuffer::new(contract.binding())
  let first = sender.encode(contract)
  let wire = FixedArray::make(first.length(), b'\x00')
  first.copy_to(wire, destination_offset=0)
  let receiver = TensorParallelRankConfigureFrameBuffer::new(contract.binding())
  let decoded = receiver.load(wire, wire.length()).contract()
  assert_eq(decoded.digest(), contract.digest())
  assert_eq(decoded.length(), contract.length())
  ignore(sender.encode(contract))
  try first.length() catch {
    StaleFrame => ()
    _ => fail("unexpected stale configure view rejection")
  } noraise {
    _ => fail("stale configure view remained live")
  }
}

///|
test "rank configure rejects corruption reserved fields and generation replay" {
  let contract = configure_test_contract()
  let wire = FixedArray::make(contract.length(), b'\x00')
  contract.copy_to(wire, destination_offset=0)
  wire[contract.length() - 1] = wire[contract.length() - 1] ^ b'\x01'
  try decode_rank_configure(wire, wire.length(), contract.binding()) catch {
    Invalid(Checksum) => ()
    _ => fail("unexpected corrupt configure rejection")
  } noraise {
    _ => fail("corrupt configure was admitted")
  }
  contract.copy_to(wire, destination_offset=0)
  wire[72] = b'\x01'
  try decode_rank_configure(wire, wire.length(), contract.binding()) catch {
    Invalid(Reserved) => ()
    _ => fail("unexpected reserved configure rejection")
  } noraise {
    _ => fail("nonzero reserved field was admitted")
  }
  let replay_binding = configure_test_binding(
    0,
    contract.startup(),
    group_generation=10UL,
  )
  try decode_rank_configure(wire, wire.length(), replay_binding) catch {
    Invalid(Reserved) => ()
    _ => fail("unexpected precedence after reserved corruption")
  } noraise {
    _ => fail("reserved corruption was hidden by binding substitution")
  }
  contract.copy_to(wire, destination_offset=0)
  try decode_rank_configure(wire, wire.length(), replay_binding) catch {
    Invalid(GroupGeneration) => ()
    _ => fail("unexpected prior-generation replay rejection")
  } noraise {
    _ => fail("prior-generation Configure replay was admitted")
  }
}

///|
test "rank configure rejects nested source device and binding substitution" {
  let source = configure_test_source()
  let startup = configure_test_startup(0, source)
  let binding = configure_test_binding(0, startup)
  let envelope = configure_test_envelope(0, startup)
  let foreign_source = configure_test_source()
  let foreign_startup = try! @worker_wire.WorkerStartupContract::new(
    identity=startup.identity(),
    bootstrap_digest=startup.bootstrap_digest(),
    bootstrap_source_digest=@worker_wire.WorkerBootstrapSourceDigest::from_sha256(
      "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
    ),
    device_ordinal=startup.device_ordinal(),
    model_generation_value=startup.model_generation_value(),
    worker_limits=startup.worker_limits(),
    inference_limits=startup.inference_limits(),
    has_predecessor=startup.has_predecessor(),
    predecessor_value=startup.predecessor_value(),
  )
  try
    compose_rank_configure(
      binding,
      foreign_startup,
      foreign_source,
      configure_test_topology(),
      envelope,
    )
  catch {
    Invalid(BootstrapSource) => ()
    _ => fail("unexpected bootstrap-source substitution rejection")
  } noraise {
    _ => fail("substituted bootstrap-source digest was admitted")
  }
  try
    compose_rank_configure(
      binding,
      startup,
      source,
      configure_test_topology(swapped=true),
      envelope,
    )
  catch {
    Invalid(Topology) => ()
    _ => fail("unexpected topology ordinal substitution rejection")
  } noraise {
    _ => fail("substituted topology ordinal was admitted")
  }
  let rank_one = configure_test_envelope(1, configure_test_startup(1, source))
  try
    compose_rank_configure(
      binding,
      startup,
      source,
      configure_test_topology(),
      rank_one,
    )
  catch {
    Invalid(Rank) => ()
    _ => fail("unexpected rank-envelope substitution rejection")
  } noraise {
    _ => fail("substituted rank envelope was admitted")
  }
}
