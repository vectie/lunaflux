///|
fn luna_paged_dtype_tag(dtype : @spec.ModelDType) -> Int {
  match dtype {
    BFloat16 => 1
    Float16 => 2
    Float32 => 3
  }
}

///|
fn LunaKernelProfileEncoder::append_digest(
  self : LunaKernelProfileEncoder,
  value : LunaKernelProfileDigest,
) -> Unit raise LunaKernelProfileError {
  self.append_string(value.as_hex())
}

///|
fn luna_paged_runtime_input_tag(value : @plan.OperationRuntimeInput) -> Int {
  match value {
    StepCounts => 1
    QueryPositions => 2
    QueryRowOffsets => 3
    SequenceLengths => 4
    PageTableOffsets => 5
    PageTablePageIndices => 6
  }
}

///|
fn LunaKernelProfileEncoder::append_paged_operand_role(
  self : LunaKernelProfileEncoder,
  role : @launch_contract.AotOperandRole,
) -> Unit raise LunaKernelProfileError {
  match role {
    ExternalInput(TokenIds) => self.append_int(1)
    WeightInput(tensor) => {
      self.append_int(2)
      self.append_int(tensor.as_int())
    }
    WeightScaleInput(tensor, scale) => {
      self.append_int(3)
      self.append_int(tensor.as_int())
      self.append_int(scale.tensor().as_int())
    }
    ActivationInput(activation) => {
      self.append_int(4)
      self.append_int(activation.as_int())
    }
    RuntimeInput(input) => {
      self.append_int(5)
      self.append_int(luna_paged_runtime_input_tag(input))
    }
    PersistentKvComponent(layer, component) => {
      self.append_int(6)
      self.append_int(layer.as_int())
      self.append_int(
        match component {
          Key => 1
          Value => 2
        },
      )
    }
    ActivationOutput(activation) => {
      self.append_int(7)
      self.append_int(activation.as_int())
    }
    Workspace => self.append_int(8)
  }
}

///|
fn LunaKernelProfileEncoder::append_paged_launch(
  self : LunaKernelProfileEncoder,
  launch : @launch_contract.PagedKvAotLaunchContract,
) -> Unit raise LunaKernelProfileError {
  self.append_int(launch.profile().id().as_int())
  self.append_int(launch.operation_id().as_int())
  self.append_int(launch.capability().as_int())
  self.append_int(launch.semantic_version().as_int())
  match launch.kv_layer() {
    None => self.append_int(-1)
    Some(layer) => self.append_int(layer.as_int())
  }
  self.append_int(launch.runtime_inputs().length())
  for input in launch.runtime_inputs() {
    self.append_int(luna_paged_runtime_input_tag(input))
  }
  match launch.persistent_state() {
    NoPersistentState => self.append_int(0)
    KvCacheReadWrite(layer) => {
      self.append_int(1)
      self.append_int(layer.as_int())
    }
  }
  self.append_int(launch.kv_layout_version())
  self.append_int(launch.tokens_per_page())
  let entry = launch.entry_point()
  self.append_string(entry.family().module_digest().sha256())
  self.append_int(entry.family().family_id().as_int())
  self.append_int(entry.entry_point_id().as_int())
  let dimensions = launch.dimensions()
  self.append_int(dimensions.grid_x())
  self.append_int(dimensions.grid_y())
  self.append_int(dimensions.grid_z())
  self.append_int(dimensions.block_x())
  self.append_int(dimensions.block_y())
  self.append_int(dimensions.block_z())
  self.append_int(dimensions.shared_memory_bytes())
  self.append_int(launch.operands().length())
  for operand in launch.operands() {
    self.append_paged_operand_role(operand.role())
    self.append_i64(operand.byte_count())
    self.append_i64(operand.alignment())
  }
  self.append_i64(launch.workspace_bytes())
  self.append_i64(launch.workspace_alignment())
}

///|
fn digest_luna_paged_launches(
  launches : ArrayView[@launch_contract.PagedKvAotLaunchContract],
) -> LunaKernelProfileDigest raise LunaKernelProfileError {
  let encoder : LunaKernelProfileEncoder = { bytes: [], maximum: 4_194_304 }
  encoder.append_string("lunaflux.kernel-paged-launch-identity.v1")
  encoder.append_int(launches.length())
  for launch in launches {
    encoder.append_paged_launch(launch)
  }
  LunaKernelProfileDigest(
    @crypto.bytes_to_hex_string(
      @crypto.sha256(Bytes::from_array(encoder.bytes)),
    ),
  )
}

///|
fn LunaKernelProfileEncoder::append_paged_contract(
  self : LunaKernelProfileEncoder,
  contract : LunaPagedProfileContract,
) -> Unit raise LunaKernelProfileError {
  let profile = contract.kernel_profile()
  let layout = contract.device_kv_layout()
  let geometry = layout.geometry()
  self.append_int(
    match contract.scope() {
      KvSubsequence => 1
      FullGraph => 2
    },
  )
  self.append_int(contract.catalog_version().as_int())
  self.append_int(profile.id().as_int())
  self.append_int(profile.max_query_rows())
  self.append_int(profile.max_query_tokens())
  self.append_int(profile.max_page_table_entries())
  self.append_int(geometry.layer_count())
  self.append_int(geometry.layout().stable_version())
  self.append_int(layout.stable_layout_version())
  self.append_int(layout.tokens_per_page())
  self.append_int(layout.total_page_count())
  self.append_int(geometry.kv_head_count())
  self.append_int(geometry.head_dimension())
  self.append_int(luna_paged_dtype_tag(geometry.dtype()))
  self.append_i64(layout.segment_alignment_bytes())
  self.append_i64(layout.page_logical_bytes())
  self.append_i64(layout.page_stride_bytes())
  self.append_i64(layout.component_stride_bytes())
  self.append_i64(layout.layer_stride_bytes())
  self.append_i64(layout.arena_bytes())
  self.append_i64(layout.limits().max_arena_bytes())
  self.append_digest(contract.launch_identity())
  self.append_int(contract.launches().length())
  for launch in contract.launches() {
    self.append_paged_launch(launch)
  }
}

///|
fn LunaKernelProfileEncoder::append_paged_shape(
  self : LunaKernelProfileEncoder,
  tokens_per_page : Int,
  prefill_row_count : Int,
  decode_row_count : Int,
  query_token_count : Int,
  page_table_block_count : Int,
  page_table_trace_digest : LunaPagedPageTableTraceDigest,
  rows : ArrayView[LunaPagedProfileRow],
) -> Unit raise LunaKernelProfileError {
  self.append_int(tokens_per_page)
  self.append_int(prefill_row_count)
  self.append_int(decode_row_count)
  self.append_int(query_token_count)
  self.append_int(page_table_block_count)
  self.append_string(page_table_trace_digest.as_hex())
  self.append_int(rows.length())
  for row in rows {
    self.append_int(row.phase().stable_int())
    self.append_int(row.query_length())
    self.append_int(row.context_length())
    self.append_int(row.cache_position())
    self.append_int(row.page_table_start())
    self.append_int(row.block_count())
  }
}

///|
fn digest_luna_paged_profile_shape(
  tokens_per_page : Int,
  prefill_row_count : Int,
  decode_row_count : Int,
  query_token_count : Int,
  page_table_block_count : Int,
  page_table_trace_digest : LunaPagedPageTableTraceDigest,
  rows : ArrayView[LunaPagedProfileRow],
) -> LunaKernelProfileDigest raise LunaKernelProfileError {
  let encoder : LunaKernelProfileEncoder = { bytes: [], maximum: 1_048_576 }
  encoder.append_string("lunaflux.kernel-paged-profile-shape.v1")
  encoder.append_paged_shape(
    tokens_per_page, prefill_row_count, decode_row_count, query_token_count, page_table_block_count,
    page_table_trace_digest, rows,
  )
  LunaKernelProfileDigest(
    @crypto.bytes_to_hex_string(
      @crypto.sha256(Bytes::from_array(encoder.bytes)),
    ),
  )
}

///|
fn encode_luna_paged_kernel_profile(
  identity : @spec.ModelIdentity,
  target : @catalog.DeviceTarget,
  workload : LunaKernelProfileDigest,
  profiler : LunaKernelProfileDigest,
  trace : LunaKernelProfileDigest,
  contract : LunaPagedProfileContract,
  sampled_wall_time_ns : Int64,
  attributed_self_time_ns : Int64,
  observations : ArrayView[LunaPagedKernelProfileObservation],
  maximum : Int,
) -> Bytes raise LunaKernelProfileError {
  let encoder : LunaKernelProfileEncoder = { bytes: [], maximum }
  encoder.append_string("lunaflux.kernel-paged-profile-priority.v1")
  encoder.append_string(identity.content().as_hex())
  encoder.append_string(identity.plan().as_hex())
  encoder.append_int(target.compute_major())
  encoder.append_int(target.compute_minor())
  encoder.append_int(if target.supports_bf16() { 1 } else { 0 })
  encoder.append_int(if target.supports_cublas_lt() { 1 } else { 0 })
  encoder.append_digest(workload)
  encoder.append_digest(profiler)
  encoder.append_digest(trace)
  encoder.append_paged_contract(contract)
  encoder.append_i64(sampled_wall_time_ns)
  encoder.append_i64(attributed_self_time_ns)
  encoder.append_int(observations.length())
  for observation in observations {
    let shape = observation.shape()
    encoder.append_int(observation.operation_id().as_int())
    encoder.append_digest(shape.digest())
    encoder.append_paged_shape(
      shape.tokens_per_page(),
      shape.prefill_row_count(),
      shape.decode_row_count(),
      shape.query_token_count(),
      shape.page_table_block_count(),
      shape.page_table_trace_digest(),
      shape.rows(),
    )
    encoder.append_i64(observation.call_count())
    encoder.append_i64(observation.self_time_ns())
    encoder.append_i64(observation.inclusive_time_ns())
    encoder.append_int(observation.counters().length())
    for counter in observation.counters() {
      encoder.append_int(counter.kind().stable_int())
      encoder.append_i64(counter.value())
    }
  }
  Bytes::from_array(encoder.bytes)
}
