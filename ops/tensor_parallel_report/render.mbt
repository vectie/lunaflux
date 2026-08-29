///|
fn append_int(output : StringBuilder, label : String, value : Int) -> Unit {
  output.write_string(label)
  output.write_string(": ")
  output.write_string(value.to_string())
  output.write_string("\n")
}

///|
fn append_int64(output : StringBuilder, label : String, value : Int64) -> Unit {
  output.write_string(label)
  output.write_string(": ")
  output.write_string(value.to_string())
  output.write_string("\n")
}

///|
fn append_bool(output : StringBuilder, label : String, value : Bool) -> Unit {
  output.write_string(label)
  output.write_string(": ")
  output.write_string(if value { "true" } else { "false" })
  output.write_string("\n")
}

///|
fn append_rank(
  output : StringBuilder,
  rank : @tensor_parallel_admission.TensorParallelRankCapacity,
) -> Unit {
  let prefix = "rank.\{rank.rank()}."
  append_int(output, prefix + "world_size", rank.world_size())
  append_int(output, prefix + "process_ordinal", rank.process_ordinal())
  output.write_string(prefix)
  output.write_string("device_name: ")
  output.write_string(rank.device_name())
  output.write_string("\n")
  append_int(output, prefix + "peer_rank_count", rank.peer_rank_count())
  append_int64(output, prefix + "memory.physical_bytes", rank.physical_bytes())
  append_int64(
    output,
    prefix + "memory.ceiling_bytes",
    rank.memory_ceiling_bytes(),
  )
  append_int64(
    output,
    prefix + "memory.weight_reserved_bytes",
    rank.weight_reserved_bytes(),
  )
  append_int64(
    output,
    prefix + "memory.activation_workspace_reserved_bytes",
    rank.activation_workspace_reserved_bytes(),
  )
  append_int64(
    output,
    prefix + "memory.kv_reserved_bytes",
    rank.kv_reserved_bytes(),
  )
  append_int64(
    output,
    prefix + "memory.collective_reserved_bytes",
    rank.collective_reserved_bytes(),
  )
  append_int64(
    output,
    prefix + "memory.reserved_total_bytes",
    rank.reserved_total_bytes(),
  )
  append_int64(
    output,
    prefix + "memory.available_bytes",
    rank.available_bytes(),
  )
}

///|
/// Renders fixed-order same-host topology and exact capacity evidence.
pub fn render(report : TensorParallelCapacityReport) -> String {
  let admission = report.admission
  let target = admission.target()
  let aggregate = admission.aggregate_capacity()
  let output = StringBuilder()
  output.write_string("LunaFlux admitted local tensor-parallel capacity\n")
  output.write_string("support.scope: same_host_tensor_parallel\n")
  output.write_string("support.cross_node: rejected\n")
  output.write_string("support.pipeline_parallel: rejected\n")
  output.write_string("support.heterogeneous_targets: rejected\n")
  append_int(output, "topology.driver_version", admission.driver_version())
  append_int(output, "topology.world_size", admission.world_size())
  append_int(output, "topology.peer_link_count", admission.peer_link_count())
  append_int(output, "topology.target.compute_major", target.compute_major())
  append_int(output, "topology.target.compute_minor", target.compute_minor())
  append_bool(output, "topology.target.bf16", target.supports_bf16())
  append_bool(output, "topology.target.cublas_lt", target.supports_cublas_lt())
  append_int64(
    output,
    "aggregate.memory.physical_bytes",
    aggregate.physical_bytes(),
  )
  append_int64(
    output,
    "aggregate.memory.ceiling_bytes",
    aggregate.memory_ceiling_bytes(),
  )
  append_int64(
    output,
    "aggregate.memory.weight_reserved_bytes",
    aggregate.weight_reserved_bytes(),
  )
  append_int64(
    output,
    "aggregate.memory.activation_workspace_reserved_bytes",
    aggregate.activation_workspace_reserved_bytes(),
  )
  append_int64(
    output,
    "aggregate.memory.kv_reserved_bytes",
    aggregate.kv_reserved_bytes(),
  )
  append_int64(
    output,
    "aggregate.memory.collective_reserved_bytes",
    aggregate.collective_reserved_bytes(),
  )
  append_int64(
    output,
    "aggregate.memory.reserved_total_bytes",
    aggregate.reserved_total_bytes(),
  )
  append_int64(
    output,
    "aggregate.memory.available_bytes",
    aggregate.available_bytes(),
  )
  append_int64(
    output,
    "aggregate.memory.minimum_rank_available_bytes",
    aggregate.minimum_rank_available_bytes(),
  )
  for rank in admission.rank_capacities() {
    append_rank(output, rank)
  }
  output.to_string()
}

///|
fn area_name(
  area : @tensor_parallel_admission.TensorParallelAdmissionArea,
) -> String {
  match area {
    Declaration => "declaration"
    Support => "support"
    Topology => "topology"
    RankBinding => "rank_binding"
    RankMemory => "rank_memory"
    Aggregate => "aggregate"
  }
}

///|
fn reason_name(
  reason : @tensor_parallel_admission.TensorParallelAdmissionReason,
) -> String {
  match reason {
    InvalidCount => "invalid_count"
    InvalidRank => "invalid_rank"
    InvalidWorld => "invalid_world"
    InvalidOrdinal => "invalid_ordinal"
    InvalidDeviceIdentity => "invalid_device_identity"
    InvalidPeerCount => "invalid_peer_count"
    InvalidMemory => "invalid_memory"
    CrossNodeUnsupported => "cross_node_unsupported"
    PipelineUnsupported => "pipeline_unsupported"
    CountMismatch => "count_mismatch"
    RankMismatch => "rank_mismatch"
    WorldMismatch => "world_mismatch"
    OrdinalMismatch => "ordinal_mismatch"
    DeviceIdentityMismatch => "device_identity_mismatch"
    TargetMismatch => "target_mismatch"
    IncompletePeerLinks => "incomplete_peer_links"
    CeilingExceedsPhysical => "ceiling_exceeds_physical"
    ReservationExceedsCeiling => "reservation_exceeds_ceiling"
    Overflow => "overflow"
  }
}

///|
/// Renders a bounded startup rejection without user or vendor payloads.
pub fn render_rejection(
  error : @tensor_parallel_admission.TensorParallelAdmissionError,
) -> String {
  match error {
    Rejected(area~, reason~, rank~) => {
      let output = StringBuilder()
      output.write_string("tensor_parallel.admission: rejected\n")
      output.write_string("rejection.area: ")
      output.write_string(area_name(area))
      output.write_string("\n")
      output.write_string("rejection.reason: ")
      output.write_string(reason_name(reason))
      output.write_string("\n")
      append_int(output, "rejection.rank", rank)
      output.to_string()
    }
  }
}
