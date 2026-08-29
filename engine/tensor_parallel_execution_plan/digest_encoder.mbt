///|
priv struct ExecutionDigestEncoder {
  bytes : Array[Byte]
}

///|
fn ExecutionDigestEncoder::append_u64(
  self : ExecutionDigestEncoder,
  value : UInt64,
) -> Unit {
  for byte in value.to_le_bytes() {
    self.bytes.push(byte)
  }
}

///|
fn ExecutionDigestEncoder::append_int(
  self : ExecutionDigestEncoder,
  value : Int,
) -> Unit {
  self.append_u64(value.to_int64().reinterpret_as_uint64())
}

///|
fn ExecutionDigestEncoder::append_i64(
  self : ExecutionDigestEncoder,
  value : Int64,
) -> Unit {
  self.append_u64(value.reinterpret_as_uint64())
}

///|
fn ExecutionDigestEncoder::append_bool(
  self : ExecutionDigestEncoder,
  value : Bool,
) -> Unit {
  self.append_int(if value { 1 } else { 0 })
}

///|
fn ExecutionDigestEncoder::append_string(
  self : ExecutionDigestEncoder,
  value : StringView,
) -> Unit {
  let bytes = @utf8.encode(value)
  self.append_int(bytes.length())
  for byte in bytes {
    self.bytes.push(byte)
  }
}

///|
fn ExecutionDigestEncoder::append_identity(
  self : ExecutionDigestEncoder,
  identity : @spec.ModelIdentity,
) -> Unit {
  self.append_string(identity.content().as_hex())
  self.append_string(identity.plan().as_hex())
}

///|
fn ExecutionDigestEncoder::append_target(
  self : ExecutionDigestEncoder,
  target : @catalog.DeviceTarget,
) -> Unit {
  self.append_int(target.compute_major())
  self.append_int(target.compute_minor())
  self.append_bool(target.supports_bf16())
  self.append_bool(target.supports_cublas_lt())
}

///|
fn ExecutionDigestEncoder::append_profile(
  self : ExecutionDigestEncoder,
  profile : @launch_contract.PagedKernelProfile,
) -> Unit {
  self.append_int(profile.id().as_int())
  self.append_int(profile.max_query_rows())
  self.append_int(profile.max_query_tokens())
  self.append_int(profile.max_page_table_entries())
}

///|
fn operation_kind_code(kind : @plan.OperationKind) -> Int {
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
fn runtime_input_code(value : @plan.OperationRuntimeInput) -> Int {
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
fn component_code(value : @device_layout.DeviceKvComponent) -> Int {
  match value {
    Key => 1
    Value => 2
  }
}

///|
fn placement_code(value : @tensor_parallel_plan.TensorPlacement) -> Int {
  match value {
    Replicated => 1
    ShardRows => 2
    ShardColumns => 3
  }
}

///|
fn ExecutionDigestEncoder::append_operand_role(
  self : ExecutionDigestEncoder,
  role : @tensor_parallel_launch_contract.TensorParallelOperandRole,
) -> Unit {
  match role {
    RuntimeInput(value) => {
      self.append_int(1)
      self.append_int(runtime_input_code(value))
    }
    RankConstants(version~) => {
      self.append_int(2)
      self.append_int(version)
    }
    ExternalInput(TokenIds) => self.append_int(3)
    WeightInput(reference) => {
      self.append_int(4)
      self.append_int(reference.as_int())
    }
    ActivationInput(reference) => {
      self.append_int(5)
      self.append_int(reference.as_int())
    }
    ActivationOutput(reference) => {
      self.append_int(6)
      self.append_int(reference.as_int())
    }
    PersistentKvComponent(layer, component) => {
      self.append_int(7)
      self.append_int(layer.as_int())
      self.append_int(component_code(component))
    }
    Workspace => self.append_int(8)
  }
}

///|
fn memory_space_code(value : TensorParallelMemorySpace) -> Int {
  match value {
    DescriptorSpace => 1
    RankConstantsSpace => 2
    WeightSpace => 3
    PersistentKvSpace => 4
    ActivationWorkspaceSpace => 5
  }
}

///|
fn ExecutionDigestEncoder::append_region(
  self : ExecutionDigestEncoder,
  region : TensorParallelScalarRegion,
) -> Unit {
  self.append_int(memory_space_code(region.space()))
  self.append_i64(region.offset_bytes())
  self.append_i64(region.byte_count())
  self.append_i64(region.reserved_bytes())
  self.append_i64(region.alignment())
}

///|
fn ExecutionDigestEncoder::append_dimensions(
  self : ExecutionDigestEncoder,
  dimensions : @launch_contract.AotLaunchDimensions,
) -> Unit {
  self.append_int(dimensions.grid_x())
  self.append_int(dimensions.grid_y())
  self.append_int(dimensions.grid_z())
  self.append_int(dimensions.block_x())
  self.append_int(dimensions.block_y())
  self.append_int(dimensions.block_z())
  self.append_int(dimensions.shared_memory_bytes())
}

///|
fn ExecutionDigestEncoder::append_entry_point(
  self : ExecutionDigestEncoder,
  entry : @catalog.AotKernelEntryPoint,
) -> Unit {
  self.append_string(entry.family().module_digest().sha256())
  self.append_int(entry.family().family_id().as_int())
  self.append_int(entry.entry_point_id().as_int())
}

///|
fn ExecutionDigestEncoder::finish(
  self : ExecutionDigestEncoder,
) -> TensorParallelExecutionPlanDigest {
  {
    sha256: @crypto.bytes_to_hex_string(
      @crypto.sha256(Bytes::from_array(self.bytes)),
    ),
  }
}
