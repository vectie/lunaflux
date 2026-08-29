///|
fn ownership_source() -> @worker_wire.EncodedBootstrapSource raise {
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
      manifest_locator="manifest.json",
      manifest_digest_sha256="bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
    ),
    @worker_wire.WorkerBootstrapExecution::new(
      compute_major=8,
      compute_minor=0,
      supports_bf16=true,
      supports_cublas_lt=true,
      tokens_per_page=16,
      total_page_count=128,
    ),
    @worker_wire.WorkerBootstrapCeilings::new(
      max_model_config_bytes=4096,
      max_weight_file_bytes=1048576L,
      max_weight_arena_bytes=1048576L,
      max_activation_arena_bytes=1048576L,
      max_kv_arena_bytes=1048576L,
      max_execution_manifest_bytes=4096L,
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
          device_ordinal=0,
          memory_ceiling_bytes=1073741824L,
          weight_reserved_bytes=1048576L,
          activation_workspace_reserved_bytes=1048576L,
          kv_reserved_bytes=1048576L,
          collective_reserved_bytes=0L,
        ),
        @worker_wire.TensorParallelRankRuntimeClaim::new(
          rank=1,
          device_ordinal=1,
          memory_ceiling_bytes=1073741824L,
          weight_reserved_bytes=1048576L,
          activation_workspace_reserved_bytes=1048576L,
          kv_reserved_bytes=1048576L,
          collective_reserved_bytes=0L,
        ),
      ],
      minimum_collective_runtime_version_code=21403,
      maximum_collective_runtime_version_code=21403,
      collective_exchange_timeout_millis=600000,
    ),
  )
}

///|
fn ownership_rank_startup(
  rank : Int,
  source : @worker_wire.EncodedBootstrapSource,
) -> @worker_wire.WorkerStartupContract raise {
  @worker_wire.WorkerStartupContract::new(
    identity=ownership_identity(),
    bootstrap_digest=@worker_wire.WorkerBootstrapDigest::from_sha256(
      if rank == 0 {
        "00112233445566778899aabbccddeeff00112233445566778899aabbccddeeff"
      } else {
        "10112233445566778899aabbccddeeff00112233445566778899aabbccddeeff"
      },
    ),
    bootstrap_source_digest=source.digest(),
    device_ordinal=rank,
    model_generation_value=1UL,
    worker_limits=ownership_worker_limits(),
    inference_limits=ownership_inference_limits(),
    has_predecessor=false,
    predecessor_value=0UL,
  )
}

///|
fn ownership_topology() -> @device_topology.LocalTopologyDeclaration raise {
  let target = @catalog.DeviceTarget::new(
    compute_major=8,
    compute_minor=0,
    supports_bf16=true,
    supports_cublas_lt=true,
  )
  @device_topology.LocalTopologyDeclaration::new(
    [
      @device_topology.LocalTopologyRank::new(
        ordinal=0,
        name="rank-zero",
        total_memory_bytes=2147483648L,
        target~,
      ),
      @device_topology.LocalTopologyRank::new(
        ordinal=1,
        name="rank-one",
        total_memory_bytes=2147483648L,
        target~,
      ),
    ],
    [@device_topology.LocalTopologyLink::direct(lower_rank=0, upper_rank=1)],
  )
}

///|
fn ownership_binding(rank : Int) -> @rank_group_wire.RankGroupWireBinding raise {
  let startup = ownership_rank_startup(rank, ownership_source())
  let worker = @tensor_parallel_worker_bootstrap.worker_contract_digest(startup)
  @rank_group_wire.RankGroupWireBinding::new(
    identity=ownership_identity(),
    model_generation_value=1UL,
    group_generation_value=1UL,
    rank~,
    world_size=2,
    group_digest="1111111111111111111111111111111111111111111111111111111111111111",
    topology_digest="2222222222222222222222222222222222222222222222222222222222222222",
    worker_digest=worker.as_sha256(),
    device_plan_digest="4444444444444444444444444444444444444444444444444444444444444444",
    execution_plan_digest="9999999999999999999999999999999999999999999999999999999999999999",
    collective_digest="5555555555555555555555555555555555555555555555555555555555555555",
    kv_digest="6666666666666666666666666666666666666666666666666666666666666666",
  )
}

///|
fn ownership_write_u32(
  bytes : FixedArray[Byte],
  offset : Int,
  value : UInt,
) -> Unit {
  bytes[offset] = value.to_byte()
  bytes[offset + 1] = (value >> 8).to_byte()
  bytes[offset + 2] = (value >> 16).to_byte()
  bytes[offset + 3] = (value >> 24).to_byte()
}

///|
fn ownership_write_i32(
  bytes : FixedArray[Byte],
  offset : Int,
  value : Int,
) -> Unit {
  ownership_write_u32(bytes, offset, value.reinterpret_as_uint())
}

///|
fn ownership_write_u64(
  bytes : FixedArray[Byte],
  offset : Int,
  value : UInt64,
) -> Unit {
  for index in 0..<8 {
    bytes[offset + index] = (value >> (index * 8)).to_byte()
  }
}

///|
fn ownership_write_digest(
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
fn ownership_envelope(
  rank : Int,
  _startup : @worker_wire.WorkerStartupContract,
) -> @tensor_parallel_worker_bootstrap.TensorParallelRankBootstrapEnvelope raise {
  let binding = ownership_binding(rank)
  let bytes = FixedArray::make(752, b'\x00')
  ownership_write_u32(bytes, 0, 0x4254504c)
  ownership_write_u32(bytes, 4, 3)
  ownership_write_i32(bytes, 8, 752)
  ownership_write_i32(bytes, 16, rank)
  ownership_write_i32(bytes, 20, 2)
  ownership_write_i32(bytes, 24, rank)
  ownership_write_i32(bytes, 28, 21403)
  ownership_write_u64(bytes, 32, 1UL)
  ownership_write_u64(bytes, 40, 1UL)
  ownership_write_digest(bytes, 48, ownership_identity().content().as_hex())
  ownership_write_digest(bytes, 112, ownership_identity().plan().as_hex())
  ownership_write_digest(bytes, 176, binding.group_digest())
  ownership_write_digest(bytes, 240, binding.topology_digest())
  ownership_write_digest(bytes, 304, binding.worker_digest())
  ownership_write_digest(bytes, 368, binding.device_plan_digest())
  ownership_write_digest(
    bytes, 432, "9999999999999999999999999999999999999999999999999999999999999999",
  )
  ownership_write_digest(bytes, 496, binding.collective_digest())
  ownership_write_digest(bytes, 560, binding.kv_digest())
  for index in 0..<128 {
    bytes[624 + index] = b'\x5a'
  }
  let mut checksum : UInt = 2166136261
  for index in 0..<752 {
    if index < 12 || index >= 16 {
      checksum = (checksum ^ bytes[index].to_uint()) * 16777619
    }
  }
  ownership_write_u32(bytes, 12, checksum)
  @tensor_parallel_worker_bootstrap.TensorParallelRankEnvelopeFrameBuffer::new()
  .load(bytes, 752)
  .envelope()
}

///|
fn ownership_configure(
  rank : Int,
) -> @tensor_parallel_rank_configure.TensorParallelRankConfigureContract raise {
  let source = ownership_source()
  let startup = ownership_rank_startup(rank, source)
  @tensor_parallel_rank_configure.compose_rank_configure(
    ownership_binding(rank),
    startup,
    source,
    ownership_topology(),
    ownership_envelope(rank, startup),
  )
}
