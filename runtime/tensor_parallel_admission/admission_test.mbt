///|
fn tp_target(minor? : Int = 0) -> @catalog.DeviceTarget {
  try! @catalog.DeviceTarget::new(
    compute_major=8,
    compute_minor=minor,
    supports_bf16=true,
    supports_cublas_lt=true,
  )
}

///|
fn tp_topology(
  physical_bytes? : Int64 = 8589934592L,
) -> @device_topology.AdmittedLocalTopology {
  let target = tp_target()
  let ranks = [
    try! @device_topology.LocalTopologyRank::new(
      ordinal=2,
      name="gpu-two",
      total_memory_bytes=physical_bytes,
      target~,
    ),
    try! @device_topology.LocalTopologyRank::new(
      ordinal=5,
      name="gpu-five",
      total_memory_bytes=physical_bytes,
      target~,
    ),
  ]
  let declaration = try! @device_topology.LocalTopologyDeclaration::new(ranks, [
    try! @device_topology.LocalTopologyLink::direct(lower_rank=0, upper_rank=1),
  ])
  let projection = try! @device_topology.LocalTopologyProbeProjection::from_observations(
    driver_version=12080,
    [
      try! @device_topology.LocalProbeDevice::new(
        ordinal=2,
        name="gpu-two",
        total_memory_bytes=physical_bytes,
        target~,
      ),
      try! @device_topology.LocalProbeDevice::new(
        ordinal=5,
        name="gpu-five",
        total_memory_bytes=physical_bytes,
        target~,
      ),
    ],
    [
      try! @device_topology.LocalPeerAccess::new(
        source_ordinal=2,
        destination_ordinal=5,
      ),
      try! @device_topology.LocalPeerAccess::new(
        source_ordinal=5,
        destination_ordinal=2,
      ),
    ],
  )
  try! @device_topology.admit_local_topology(declaration, projection)
}

///|
fn tp_claim(
  rank : Int,
  world_size? : Int = 2,
  process_ordinal? : Int = -1,
  device_name? : String = "",
  target? : @catalog.DeviceTarget = tp_target(),
  peer_rank_count? : Int = 1,
  memory_ceiling_bytes? : Int64 = 6442450944L,
  weight_reserved_bytes? : Int64 = 1073741824L,
  activation_workspace_reserved_bytes? : Int64 = 536870912L,
  kv_reserved_bytes? : Int64 = 1073741824L,
  collective_reserved_bytes? : Int64 = 268435456L,
) -> TensorParallelRankDeclaration raise TensorParallelAdmissionError {
  let ordinal = if process_ordinal < 0 {
    if rank == 0 {
      2
    } else {
      5
    }
  } else {
    process_ordinal
  }
  let name = if device_name.is_empty() {
    if rank == 0 {
      "gpu-two"
    } else {
      "gpu-five"
    }
  } else {
    device_name
  }
  @tensor_parallel_admission.TensorParallelRankDeclaration::new(
    rank~,
    world_size~,
    process_ordinal=ordinal,
    device_name=name,
    target~,
    peer_rank_count~,
    memory_ceiling_bytes~,
    weight_reserved_bytes~,
    activation_workspace_reserved_bytes~,
    kv_reserved_bytes~,
    collective_reserved_bytes~,
  )
}

///|
fn tp_default_ranks() -> Array[TensorParallelRankDeclaration] {
  [try! tp_claim(0), try! tp_claim(1)]
}

///|
fn tp_declaration(
  host_count? : Int = 1,
  pipeline_stage_count? : Int = 1,
  ranks? : Array[TensorParallelRankDeclaration] = tp_default_ranks(),
) -> TensorParallelRuntimeDeclaration raise TensorParallelAdmissionError {
  @tensor_parallel_admission.TensorParallelRuntimeDeclaration::new(
    host_count~,
    pipeline_stage_count~,
    ranks,
  )
}

///|
fn capture_tp_admission_rejection(
  run : () -> TensorParallelRuntimeAdmission raise TensorParallelAdmissionError,
) -> TensorParallelAdmissionError? {
  try {
    ignore(run())
    None
  } catch {
    error => Some(error)
  }
}

///|
fn capture_tp_declaration_rejection(
  run : () -> TensorParallelRankDeclaration raise TensorParallelAdmissionError,
) -> TensorParallelAdmissionError? {
  try {
    ignore(run())
    None
  } catch {
    error => Some(error)
  }
}

///|
test "exact local rank bindings publish per-rank and aggregate capacity" {
  let second = try! tp_claim(1, weight_reserved_bytes=1610612736L)
  let admitted = try! @tensor_parallel_admission.admit(
    tp_topology(),
    try! tp_declaration(ranks=[try! tp_claim(0), second]),
  )
  assert_eq(admitted.driver_version(), 12080)
  assert_eq(admitted.world_size(), 2)
  assert_eq(admitted.peer_link_count(), 1)
  assert_eq(admitted.target(), tp_target())
  guard admitted.rank_capacity(0) is Some(first) &&
    admitted.rank_capacity(1) is Some(second) else {
    fail("admitted rank capacity was missing")
  }
  assert_eq(first.rank(), 0)
  assert_eq(first.world_size(), 2)
  assert_eq(first.process_ordinal(), 2)
  assert_eq(first.device_name(), "gpu-two")
  assert_eq(first.peer_rank_count(), 1)
  assert_eq(first.physical_bytes(), 8589934592L)
  assert_eq(first.memory_ceiling_bytes(), 6442450944L)
  assert_eq(first.reserved_total_bytes(), 2952790016L)
  assert_eq(first.available_bytes(), 3489660928L)
  assert_eq(second.reserved_total_bytes(), 3489660928L)
  assert_eq(second.available_bytes(), 2952790016L)
  let aggregate = admitted.aggregate_capacity()
  assert_eq(aggregate.physical_bytes(), 17179869184L)
  assert_eq(aggregate.memory_ceiling_bytes(), 12884901888L)
  assert_eq(aggregate.weight_reserved_bytes(), 2684354560L)
  assert_eq(aggregate.activation_workspace_reserved_bytes(), 1073741824L)
  assert_eq(aggregate.kv_reserved_bytes(), 2147483648L)
  assert_eq(aggregate.collective_reserved_bytes(), 536870912L)
  assert_eq(aggregate.reserved_total_bytes(), 6442450944L)
  assert_eq(aggregate.available_bytes(), 6442450944L)
  assert_eq(aggregate.minimum_rank_available_bytes(), 2952790016L)
}

///|
test "unsupported host and pipeline layouts fail startup distinctly" {
  for
    declaration in [
      try! tp_declaration(host_count=2),
      try! tp_declaration(pipeline_stage_count=2),
    ] {
    match
      capture_tp_admission_rejection(() => {
        @tensor_parallel_admission.admit(tp_topology(), declaration)
      }) {
      Some(Rejected(area=Support, reason=CrossNodeUnsupported, rank=-1)) => ()
      Some(Rejected(area=Support, reason=PipelineUnsupported, rank=-1)) => ()
      _ => fail("unexpected unsupported-layout rejection")
    }
  }
}

///|
test "rank scalar substitutions and heterogeneous claims fail closed" {
  let other_target = tp_target(minor=6)
  let hostile : Array[Array[TensorParallelRankDeclaration]] = [
    [try! tp_claim(1), try! tp_claim(0)],
    [try! tp_claim(0, world_size=3), try! tp_claim(1)],
    [try! tp_claim(0, process_ordinal=3), try! tp_claim(1)],
    [try! tp_claim(0, device_name="replacement"), try! tp_claim(1)],
    [try! tp_claim(0, target=other_target), try! tp_claim(1)],
    [try! tp_claim(0, peer_rank_count=0), try! tp_claim(1)],
  ]
  for ranks in hostile {
    let declaration = try! tp_declaration(ranks~)
    match
      capture_tp_admission_rejection(() => {
        @tensor_parallel_admission.admit(tp_topology(), declaration)
      }) {
      Some(Rejected(area=RankBinding, ..)) => ()
      _ => fail("unexpected scalar-substitution rejection")
    }
  }
}

///|
test "rank and aggregate memory arithmetic fail before publication" {
  let reservation_rejection = capture_tp_declaration_rejection(() => {
    @tensor_parallel_admission.TensorParallelRankDeclaration::new(
      rank=0,
      world_size=2,
      process_ordinal=2,
      device_name="gpu-two",
      target=tp_target(),
      peer_rank_count=1,
      memory_ceiling_bytes=268435456L,
      weight_reserved_bytes=268435456L,
      activation_workspace_reserved_bytes=1L,
      kv_reserved_bytes=0L,
      collective_reserved_bytes=0L,
    )
  })
  assert_eq(
    reservation_rejection,
    Some(Rejected(area=RankMemory, reason=ReservationExceedsCeiling, rank=0)),
  )
  let physical_declaration = try! tp_declaration(ranks=[
    try! tp_claim(0, memory_ceiling_bytes=9663676416L),
    try! tp_claim(1),
  ])
  let physical_rejection = capture_tp_admission_rejection(() => {
    @tensor_parallel_admission.admit(tp_topology(), physical_declaration)
  })
  assert_eq(
    physical_rejection,
    Some(Rejected(area=RankMemory, reason=CeilingExceedsPhysical, rank=0)),
  )
  let overflow_declaration = try! tp_declaration()
  let overflow_rejection = capture_tp_admission_rejection(() => {
    @tensor_parallel_admission.admit(
      tp_topology(physical_bytes=0x7fffffffffffffffL),
      overflow_declaration,
    )
  })
  assert_eq(
    overflow_rejection,
    Some(Rejected(area=Aggregate, reason=Overflow, rank=-1)),
  )
}
