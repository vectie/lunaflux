///|
priv struct GroupTemplateEncoder {
  bytes : Array[Byte]
}

///|
fn GroupTemplateEncoder::append(self : Self, value : StringView) -> Unit {
  let encoded = @utf8.encode(value.to_owned())
  for byte in @utf8.encode(encoded.length().to_string()) {
    self.bytes.push(byte)
  }
  self.bytes.push(b':')
  for byte in encoded {
    self.bytes.push(byte)
  }
}

///|
fn GroupTemplateEncoder::append_int(self : Self, value : Int) -> Unit {
  self.append(value.to_string())
}

///|
fn GroupTemplateEncoder::append_int64(self : Self, value : Int64) -> Unit {
  self.append(value.to_string())
}

///|
fn template_placement_code(
  value : @tensor_parallel_plan.TensorPlacement,
) -> Int {
  match value {
    Replicated => 1
    ShardColumns => 2
    ShardRows => 3
  }
}

///|
fn template_collective_code(
  value : @tensor_parallel_plan.TensorCollectiveKind,
) -> Int {
  match value {
    SumAllReduce => 1
    LastAxisAllGather => 2
  }
}

///|
fn derive_group_template_digest(
  source : @worker_wire.EncodedBootstrapSource,
  model_plan : @plan.ModelPlan,
  model_generation : @worker_protocol.ModelPlanGeneration,
  topology_declaration : @device_topology.LocalTopologyDeclaration,
  inspection_blueprints : ArrayView[
    @device_materialize.DeviceShardedWeightFileInspection,
  ],
  device_step_limits : @device_step.DeviceStepLimits,
  manifest_limits : @execution_manifest_file.ExecutionManifestFileLimits,
  launch_limits : @tensor_parallel_launch_contract.TensorParallelLaunchContractLimits,
  execution_limits : @tensor_parallel_execution_plan.TensorParallelExecutionPlanLimits,
  worker_limits : @worker_protocol.WorkerProtocolLimits,
  inference_limits : @inference.InferenceLimits,
  process_limits : @rank_group_process.RankGroupProcessLimits,
  admitted_rank_child_activation_path : Bytes,
  rank_child_executable_digest_sha256 : StringView,
) -> TensorParallelGroupTemplateDigest raise TensorParallelGroupTransportError {
  if rank_child_executable_digest_sha256.length() != 64 ||
    rank_child_executable_digest_sha256 == "0".repeat(64) {
    rejected(Source)
  }
  let encoder : GroupTemplateEncoder = { bytes: [] }
  encoder.append("lunaflux.tensor-parallel-group-template.v1")
  encoder.append(source.digest().as_hex())
  encoder.append(source.kernel().manifest_digest())
  match source.kernel().luna_approval() {
    Some(approval) => {
      encoder.append("external_ed25519_v1")
      encoder.append(approval.manifest_digest())
      encoder.append(approval.key_identity())
      encoder.append(approval.signature_digest())
      encoder.append(approval.approved_source())
    }
    None => encoder.append("legacy_unapproved")
  }
  encoder.append(model_plan.identity().content().as_hex())
  encoder.append(model_plan.identity().plan().as_hex())
  encoder.append(model_generation.value().to_string())
  encoder.append_int(topology_declaration.rank_count())
  for rank in 0..<topology_declaration.rank_count() {
    guard topology_declaration.rank(rank) is Some(device) else {
      rejected(Topology, rank~)
    }
    encoder.append_int(rank)
    encoder.append_int(device.ordinal())
    encoder.append(device.name())
    encoder.append_int64(device.total_memory_bytes())
    encoder.append_int(device.target().compute_major())
    encoder.append_int(device.target().compute_minor())
  }
  encoder.append_int(inspection_blueprints.length())
  for rank, inspection in inspection_blueprints {
    let shard = inspection.rank_plan()
    encoder.append_int(rank)
    encoder.append(inspection.content_digest().as_hex())
    encoder.append_int(shard.topology().rank())
    encoder.append_int(shard.topology().world_size())
    encoder.append_int64(shard.alignment_bytes())
    encoder.append_int64(shard.arena_bytes())
    encoder.append_int64(shard.materialized_bytes())
    encoder.append_int(shard.tensor_count())
    encoder.append_int(shard.operation_count())
    encoder.append_int(shard.collective_count())
    for index in 0..<model_plan.tensor_count() {
      let reference = @plan.TensorRef::new(index) catch {
        _ => rejected(RankInspection, rank~)
      }
      guard shard.tensor(reference) is Some(tensor) else {
        rejected(RankInspection, rank~)
      }
      let extent = tensor.extent()
      let region = tensor.region()
      let transfer = tensor.transfer_recipe()
      encoder.append_int(reference.as_int())
      encoder.append_int(template_placement_code(tensor.placement()))
      for
        value in [
          extent.full_rows(),
          extent.full_columns(),
          extent.row_start(),
          extent.row_count(),
          extent.column_start(),
          extent.column_count(),
          extent.byte_length(),
          region.offset_bytes(),
          region.byte_length(),
          region.alignment_bytes(),
          transfer.segment_byte_length(),
          transfer.segment_count(),
        ] {
        encoder.append_int64(value)
      }
      for segment_index in 0L..<transfer.segment_count() {
        let segment = transfer.segment(segment_index) catch {
          _ => rejected(RankInspection, rank~)
        }
        encoder.append_int64(segment.source_payload_offset_bytes())
        encoder.append_int64(segment.destination_offset_bytes())
        encoder.append_int64(segment.byte_length())
      }
    }
    for operation in model_plan.operations() {
      guard shard.operation(operation.id()) is Some(rank_operation) else {
        rejected(RankInspection, rank~)
      }
      encoder.append_int(rank_operation.operation().as_int())
      encoder.append_int(rank_operation.capability().as_int())
      for
        value in [
          rank_operation.input_width(),
          rank_operation.output_width(),
          rank_operation.inner_width(),
          rank_operation.query_head_count(),
          rank_operation.kv_head_count(),
          rank_operation.head_dimension(),
        ] {
        encoder.append_int(value)
      }
    }
    for collective_index in 0..<shard.collective_count() {
      guard shard.collective(collective_index) is Some(collective) else {
        rejected(RankInspection, rank~)
      }
      encoder.append_int(collective.sequence().as_int())
      encoder.append_int(collective.operation().as_int())
      encoder.append_int(template_collective_code(collective.kind()))
      encoder.append_int(collective.full_width())
      encoder.append_int(collective.local_width())
    }
  }
  for
    value in [
      device_step_limits.max_rows(),
      device_step_limits.max_tokens(),
      device_step_limits.max_pages(),
      device_step_limits.max_sequence_tokens(),
      device_step_limits.max_token_id(),
      manifest_limits.max_operations(),
      launch_limits.max_profiles(),
      launch_limits.max_contracts(),
      launch_limits.max_operands_per_contract(),
      execution_limits.max_operations(),
      execution_limits.max_operands(),
      execution_limits.max_descriptors(),
      execution_limits.max_collectives(),
      worker_limits.max_plan_rows(),
      worker_limits.max_plan_tokens(),
      worker_limits.max_plan_pages(),
      worker_limits.max_sequence_tokens(),
      worker_limits.max_token_id(),
      inference_limits.max_input_tokens(),
      inference_limits.max_new_tokens(),
      inference_limits.max_context_tokens(),
      inference_limits.max_token_id(),
      process_limits.max_payload_bytes(),
      process_limits.collective_exchange_timeout_millis(),
    ] {
    encoder.append_int(value)
  }
  encoder.append_int64(manifest_limits.max_manifest_bytes())
  encoder.append_int64(execution_limits.max_descriptor_bytes())
  encoder.append_int64(execution_limits.max_activation_workspace_bytes())
  let ceilings = source.ceilings()
  for
    value in [
      ceilings.max_weight_file_bytes(),
      ceilings.max_weight_arena_bytes(),
      ceilings.max_activation_arena_bytes(),
      ceilings.max_kv_arena_bytes(),
      ceilings.max_execution_manifest_bytes(),
      ceilings.max_module_bytes(),
      ceilings.max_total_module_bytes(),
    ] {
    encoder.append_int64(value)
  }
  encoder.append(rank_child_executable_digest_sha256)
  encoder.append(
    @crypto.bytes_to_hex_string(
      @crypto.sha256(admitted_rank_child_activation_path),
    ),
  )
  {
    hex: @crypto.bytes_to_hex_string(
      @crypto.sha256(Bytes::from_array(encoder.bytes)),
    ),
  }
}
