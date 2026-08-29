///|
fn report_target() -> @catalog.DeviceTarget raise @catalog.CatalogError {
  @catalog.DeviceTarget::new(
    compute_major=8,
    compute_minor=0,
    supports_bf16=true,
    supports_cublas_lt=true,
  )
}

///|
fn report_admission() -> @tensor_parallel_admission.TensorParallelRuntimeAdmission raise {
  let target = report_target()
  let topology_declaration = @device_topology.LocalTopologyDeclaration::new(
    [
      @device_topology.LocalTopologyRank::new(
        ordinal=2,
        name="gpu-two",
        total_memory_bytes=2147483648L,
        target~,
      ),
      @device_topology.LocalTopologyRank::new(
        ordinal=5,
        name="gpu-five",
        total_memory_bytes=2147483648L,
        target~,
      ),
    ],
    [@device_topology.LocalTopologyLink::direct(lower_rank=0, upper_rank=1)],
  )
  let projection = @device_topology.LocalTopologyProbeProjection::from_observations(
    driver_version=12080,
    [
      @device_topology.LocalProbeDevice::new(
        ordinal=2,
        name="gpu-two",
        total_memory_bytes=2147483648L,
        target~,
      ),
      @device_topology.LocalProbeDevice::new(
        ordinal=5,
        name="gpu-five",
        total_memory_bytes=2147483648L,
        target~,
      ),
    ],
    [
      @device_topology.LocalPeerAccess::new(
        source_ordinal=2,
        destination_ordinal=5,
      ),
      @device_topology.LocalPeerAccess::new(
        source_ordinal=5,
        destination_ordinal=2,
      ),
    ],
  )
  let topology = @device_topology.admit_local_topology(
    topology_declaration, projection,
  )
  let ranks : Array[@tensor_parallel_admission.TensorParallelRankDeclaration] = []
  for rank in 0..<2 {
    ranks.push(
      @tensor_parallel_admission.TensorParallelRankDeclaration::new(
        rank~,
        world_size=2,
        process_ordinal=if rank == 0 { 2 } else { 5 },
        device_name=if rank == 0 { "gpu-two" } else { "gpu-five" },
        target~,
        peer_rank_count=1,
        memory_ceiling_bytes=1073741824L,
        weight_reserved_bytes=268435456L,
        activation_workspace_reserved_bytes=134217728L,
        kv_reserved_bytes=268435456L,
        collective_reserved_bytes=67108864L,
      ),
    )
  }
  @tensor_parallel_admission.admit(
    topology,
    @tensor_parallel_admission.TensorParallelRuntimeDeclaration::new(
      host_count=1,
      pipeline_stage_count=1,
      ranks,
    ),
  )
}

///|
test "capacity report is deterministic complete and explicit about exclusions" {
  let report = @tensor_parallel_report.build(report_admission())
  let rendered = @tensor_parallel_report.render(report)
  assert_eq(@tensor_parallel_report.render(report), rendered)
  assert_true(rendered.contains("support.scope: same_host_tensor_parallel"))
  assert_true(rendered.contains("support.cross_node: rejected"))
  assert_true(rendered.contains("topology.world_size: 2"))
  assert_true(rendered.contains("topology.peer_link_count: 1"))
  assert_true(rendered.contains("aggregate.memory.physical_bytes: 4294967296"))
  assert_true(rendered.contains("aggregate.memory.ceiling_bytes: 2147483648"))
  assert_true(
    rendered.contains("aggregate.memory.reserved_total_bytes: 1476395008"),
  )
  assert_true(rendered.contains("aggregate.memory.available_bytes: 671088640"))
  assert_true(rendered.contains("rank.0.process_ordinal: 2"))
  assert_true(rendered.contains("rank.0.device_name: gpu-two"))
  assert_true(rendered.contains("rank.1.process_ordinal: 5"))
  assert_true(
    rendered.contains("rank.1.memory.collective_reserved_bytes: 67108864"),
  )
}

///|
test "rejection report is fixed vocabulary and payload free" {
  let rendered = @tensor_parallel_report.render_rejection(
    Rejected(area=RankMemory, reason=CeilingExceedsPhysical, rank=1),
  )
  assert_eq(
    rendered,
    (
      #|tensor_parallel.admission: rejected
      #|rejection.area: rank_memory
      #|rejection.reason: ceiling_exceeds_physical
      #|rejection.rank: 1
      #|
    ),
  )
  assert_false(rendered.contains("gpu"))
  assert_false(rendered.contains("CUDA"))
}
