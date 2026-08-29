///|
priv struct TopologyFixture {
  topology : @device_topology.AdmittedLocalTopology
  source : @worker_wire.EncodedBootstrapSource
  runtime : @device.TensorCollectiveRuntime
  rendezvous : @device.TensorCollectiveGroupId
}

///|
fn fixture_topology(
  target : @catalog.DeviceTarget,
) -> @device_topology.AdmittedLocalTopology raise {
  let ranks = [
    @device_topology.LocalTopologyRank::new(
      ordinal=0,
      name="fake-rank-zero",
      total_memory_bytes=1073741824L,
      target~,
    ),
    @device_topology.LocalTopologyRank::new(
      ordinal=1,
      name="fake-rank-one",
      total_memory_bytes=1073741824L,
      target~,
    ),
  ]
  let devices = [
    @device_topology.LocalProbeDevice::new(
      ordinal=0,
      name="fake-rank-zero",
      total_memory_bytes=1073741824L,
      target~,
    ),
    @device_topology.LocalProbeDevice::new(
      ordinal=1,
      name="fake-rank-one",
      total_memory_bytes=1073741824L,
      target~,
    ),
  ]
  @device_topology.admit_local_topology(
    @device_topology.LocalTopologyDeclaration::new(ranks, [
      @device_topology.LocalTopologyLink::direct(lower_rank=0, upper_rank=1),
    ]),
    @device_topology.LocalTopologyProbeProjection::from_observations(
      driver_version=12080,
      devices,
      [
        @device_topology.LocalPeerAccess::new(
          source_ordinal=0,
          destination_ordinal=1,
        ),
        @device_topology.LocalPeerAccess::new(
          source_ordinal=1,
          destination_ordinal=0,
        ),
      ],
    ),
  )
}

///|
fn fixture_source() -> @worker_wire.EncodedBootstrapSource raise {
  let roots = @worker_wire.WorkerBootstrapRoots::new(
    model_root="/fixture/model",
    kernel_root="/fixture/kernel",
  )
  let model = @worker_wire.WorkerBootstrapModelSource::new(
    config_locator="config.json",
    weights_locator="weights.safetensors",
    config_digest_sha256="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
  )
  let kernel = @worker_wire.WorkerBootstrapKernelSource::new(
    manifest_locator="tensor-parallel.manifest",
    manifest_digest_sha256="bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
  )
  let execution = @worker_wire.WorkerBootstrapExecution::new(
    compute_major=8,
    compute_minor=0,
    supports_bf16=true,
    supports_cublas_lt=false,
    tokens_per_page=TP_TOKENS_PER_PAGE,
    total_page_count=TP_TOTAL_PAGE_COUNT,
  )
  let ceilings = @worker_wire.WorkerBootstrapCeilings::new(
    max_model_config_bytes=1048576,
    max_weight_file_bytes=16384L,
    max_weight_arena_bytes=4096L,
    max_activation_arena_bytes=1048576L,
    max_kv_arena_bytes=1048576L,
    max_execution_manifest_bytes=1048576L,
    max_module_bytes=4096L,
    max_total_module_bytes=4096L,
  )
  let claims : Array[@worker_wire.TensorParallelRankRuntimeClaim] = []
  for rank in 0..<TP_WORLD_SIZE {
    claims.push(
      @worker_wire.TensorParallelRankRuntimeClaim::new(
        rank~,
        device_ordinal=rank,
        memory_ceiling_bytes=1073741824L,
        weight_reserved_bytes=4096L,
        activation_workspace_reserved_bytes=1048576L,
        kv_reserved_bytes=1048576L,
        collective_reserved_bytes=1048576L,
      ),
    )
  }
  @worker_wire.admit_tensor_parallel_bootstrap_source(
    roots,
    model,
    kernel,
    execution,
    ceilings,
    @worker_wire.TensorParallelBootstrapPolicy::new(
      claims,
      minimum_collective_runtime_version_code=TP_RUNTIME_VERSION,
      maximum_collective_runtime_version_code=TP_RUNTIME_VERSION,
      collective_exchange_timeout_millis=600000,
    ),
  )
}

///|
fn prepare_topology_fixture(
  target : @catalog.DeviceTarget,
) -> TopologyFixture raise {
  let topology = fixture_topology(target)
  let source = fixture_source()
  ignore(
    @tensor_parallel_worker_bootstrap.admit_runtime_policy(source, topology),
  )
  let runtime = @device.admit_tensor_collective_runtime(
    minimum_version_code=TP_RUNTIME_VERSION,
    maximum_version_code=TP_RUNTIME_VERSION,
  )
  let rendezvous = runtime.create_group_id()
  { topology, source, runtime, rendezvous }
}
