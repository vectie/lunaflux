///|
fn digest_topology(
  topology : @device_topology.AdmittedLocalTopology,
) -> TensorParallelTopologyDigest raise TensorParallelWorkerBootstrapError {
  let encoder : BootstrapDigestEncoder = { bytes: [] }
  encoder.append_string("lunaflux.tensor-parallel-topology.v1")
  encoder.append_int(topology.driver_version())
  encoder.append_int(topology.world_size())
  encoder.append_target(topology.target())
  encoder.append_int(topology.link_count())
  for rank in 0..<topology.world_size() {
    guard topology.rank(rank) is Some(device) else {
      raise rejected(IncompleteRankSet, rank)
    }
    encoder.append_int(rank)
    encoder.append_int(device.ordinal())
    encoder.append_string(device.name())
    encoder.append_int64(device.total_memory_bytes())
    encoder.append_target(device.target())
  }
  topology_digest_from_hex(encoder.finish())
}

///|
fn append_worker_limits(
  encoder : BootstrapDigestEncoder,
  limits : @worker_protocol.WorkerProtocolLimits,
) -> Unit {
  for
    value in [
      limits.max_prefill_rows(),
      limits.max_decode_rows(),
      limits.max_plan_rows(),
      limits.max_plan_tokens(),
      limits.max_plan_pages(),
      limits.max_capabilities(),
      limits.max_completion_slots(),
      limits.max_sequence_tokens(),
      limits.max_token_id(),
    ] {
    encoder.append_int(value)
  }
}

///|
fn append_inference_limits(
  encoder : BootstrapDigestEncoder,
  limits : @inference.InferenceLimits,
) -> Unit {
  for
    value in [
      limits.max_text_bytes(),
      limits.max_input_tokens(),
      limits.max_new_tokens(),
      limits.max_context_tokens(),
      limits.max_token_id(),
      limits.max_stop_token_ids(),
      limits.max_stop_strings(),
      limits.max_stop_string_bytes(),
      limits.max_trace_bytes(),
      limits.max_cache_scope_bytes(),
      limits.max_decoded_delta_bytes(),
    ] {
    encoder.append_int(value)
  }
  encoder.append_u64(limits.max_deadline_millis())
  encoder.append_int(limits.max_top_k())
  encoder.append_u64(limits.max_temperature().reinterpret_as_uint64())
}

///|
fn digest_worker_contract(
  contract : @worker_wire.WorkerStartupContract,
) -> TensorParallelWorkerContractDigest raise TensorParallelWorkerBootstrapError {
  let encoder : BootstrapDigestEncoder = { bytes: [] }
  encoder.append_string("lunaflux.tensor-parallel-worker-contract.v1")
  encoder.append_identity(contract.identity())
  encoder.append_string(contract.bootstrap_digest().as_hex())
  encoder.append_string(contract.bootstrap_source_digest().as_hex())
  encoder.append_int(contract.device_ordinal())
  encoder.append_u64(contract.model_generation_value())
  append_worker_limits(encoder, contract.worker_limits())
  append_inference_limits(encoder, contract.inference_limits())
  encoder.append_bool(contract.has_predecessor())
  encoder.append_u64(contract.predecessor_value())
  worker_digest_from_hex(encoder.finish())
}

///|
/// Returns the canonical digest of one exact worker startup contract so an
/// enclosing rank Configure frame can authenticate its outer wire binding.
pub fn worker_contract_digest(
  contract : @worker_wire.WorkerStartupContract,
) -> TensorParallelWorkerContractDigest raise TensorParallelWorkerBootstrapError {
  digest_worker_contract(contract)
}

///|
fn placement_code(placement : @tensor_parallel_plan.TensorPlacement) -> Int {
  match placement {
    Replicated => 1
    ShardRows => 2
    ShardColumns => 3
  }
}

///|
fn device_operation_kind_code(kind : @plan.OperationKind) -> Int {
  match kind {
    Embedding => 1
    RmsNorm => 2
    QkvProjection => 3
    RotaryEmbedding => 4
    CausalAttention => 5
    OutputProjection => 6
    ResidualAdd => 7
    GatedMlp => 8
    LanguageModelHead => 9
    QkRmsNorm => 10
  }
}

///|
fn append_matrix(
  encoder : BootstrapDigestEncoder,
  matrix : @tensor_parallel_device_plan.TensorParallelMatrixView,
) -> Unit {
  encoder.append_int(matrix.reference().as_int())
  encoder.append_int(placement_code(matrix.placement()))
  encoder.append_int64(matrix.full_rows())
  encoder.append_int64(matrix.full_columns())
  encoder.append_int64(matrix.local_row_start())
  encoder.append_int64(matrix.local_rows())
  encoder.append_int64(matrix.local_column_start())
  encoder.append_int64(matrix.local_columns())
  encoder.append_int64(matrix.region_offset_bytes())
  encoder.append_int64(matrix.region_byte_length())
}

///|
fn append_tensor_table(
  encoder : BootstrapDigestEncoder,
  plan : @tensor_parallel_device_plan.TensorParallelDevicePlan,
) -> Unit raise TensorParallelWorkerBootstrapError {
  encoder.append_int(plan.tensor_count())
  for index in 0..<plan.tensor_count() {
    let reference = @plan.TensorRef::new(index) catch {
      _ => raise rejected(DevicePlan, plan.rank())
    }
    guard plan.tensor(reference) is Some(matrix) else {
      raise rejected(DevicePlan, plan.rank())
    }
    append_matrix(encoder, matrix)
  }
}

///|
fn collective_kind_code(
  kind : @tensor_parallel_plan.TensorCollectiveKind,
) -> Int {
  match kind {
    SumAllReduce => 1
    LastAxisAllGather => 2
  }
}

///|
fn digest_device_plan(
  plan : @tensor_parallel_device_plan.TensorParallelDevicePlan,
) -> TensorParallelDevicePlanDigest raise TensorParallelWorkerBootstrapError {
  let encoder : BootstrapDigestEncoder = { bytes: [] }
  encoder.append_string("lunaflux.tensor-parallel-device-plan.v2")
  encoder.append_identity(plan.identity())
  encoder.append_int(plan.rank())
  encoder.append_int(plan.world_size())
  encoder.append_int(plan.device_ordinal())
  encoder.append_target(plan.target())
  encoder.append_int64(plan.arena_bytes())
  encoder.append_int64(plan.materialized_bytes())
  append_tensor_table(encoder, plan)
  encoder.append_int(plan.operation_count())
  for index in 0..<plan.operation_count() {
    let operation = @plan.OperationId::new(index) catch {
      _ => raise rejected(DevicePlan, plan.rank())
    }
    guard plan.operation(operation) is Some(geometry) else {
      raise rejected(DevicePlan, plan.rank())
    }
    encoder.append_int(geometry.operation().as_int())
    encoder.append_int(device_operation_kind_code(geometry.kind()))
    encoder.append_int(geometry.capability().as_int())
    encoder.append_int(geometry.input_width())
    encoder.append_int(geometry.output_width())
    encoder.append_int(geometry.inner_width())
    encoder.append_int(geometry.query_head_count())
    encoder.append_int(geometry.kv_head_count())
    encoder.append_int(geometry.head_dimension())
  }
  encoder.append_int(plan.collective_count())
  for index in 0..<plan.collective_count() {
    guard plan.collective(index) is Some(site) else {
      raise rejected(DevicePlan, plan.rank())
    }
    encoder.append_int(site.sequence().as_int())
    encoder.append_int(site.operation().as_int())
    encoder.append_int(collective_kind_code(site.kind()))
    encoder.append_int(site.full_width())
    encoder.append_int(site.local_width())
  }
  device_plan_digest_from_hex(encoder.finish())
}

///|
fn digest_group(
  template_digest_sha256 : StringView,
  identity : @spec.ModelIdentity,
  model_generation_value : UInt64,
  group_generation_value : UInt64,
  topology_digest : TensorParallelTopologyDigest,
  collective_runtime_version_code : Int,
  rendezvous_bytes : Bytes,
  claims : FixedArray[RankBootstrapClaim],
) -> TensorParallelGroupBootstrapDigest raise TensorParallelWorkerBootstrapError {
  let encoder : BootstrapDigestEncoder = { bytes: [] }
  encoder.append_string("lunaflux.tensor-parallel-group-bootstrap.v4")
  encoder.append_string(template_digest_sha256)
  encoder.append_identity(identity)
  encoder.append_u64(model_generation_value)
  encoder.append_u64(group_generation_value)
  encoder.append_string(topology_digest.hex)
  encoder.append_int(collective_runtime_version_code)
  encoder.append_int(claims.length())
  encoder.append_int(rendezvous_bytes.length())
  for byte in rendezvous_bytes {
    encoder.bytes.push(byte)
  }
  for claim in claims {
    encoder.append_int(claim.rank)
    encoder.append_int(claim.world_size)
    encoder.append_int(claim.device_ordinal)
    encoder.append_string(claim.worker_digest.hex)
    encoder.append_string(claim.device_plan_digest.hex)
    encoder.append_string(claim.execution_plan_digest_sha256)
    encoder.append_string(claim.collective_digest.as_hex())
    encoder.append_string(claim.kv_digest.as_sha256())
  }
  group_digest_from_hex(encoder.finish())
}
