///|
priv struct BootstrapEncoder {
  bytes : Array[Byte]
  maximum : Int
}

///|
fn BootstrapEncoder::append_byte(
  self : BootstrapEncoder,
  value : Byte,
) -> Unit raise DeviceBootstrapError {
  if self.bytes.length() >= self.maximum {
    raise InvalidBootstrap(CanonicalBytes)
  }
  self.bytes.push(value)
}

///|
fn BootstrapEncoder::append_u64(
  self : BootstrapEncoder,
  value : UInt64,
) -> Unit raise DeviceBootstrapError {
  for byte in value.to_le_bytes() {
    self.append_byte(byte)
  }
}

///|
fn BootstrapEncoder::append_int(
  self : BootstrapEncoder,
  value : Int,
) -> Unit raise DeviceBootstrapError {
  self.append_u64(Int64::from_int(value).reinterpret_as_uint64())
}

///|
fn BootstrapEncoder::append_i64(
  self : BootstrapEncoder,
  value : Int64,
) -> Unit raise DeviceBootstrapError {
  self.append_u64(value.reinterpret_as_uint64())
}

///|
fn BootstrapEncoder::append_string(
  self : BootstrapEncoder,
  value : StringView,
) -> Unit raise DeviceBootstrapError {
  let encoded = @utf8.encode(value)
  self.append_int(encoded.length())
  for byte in encoded {
    self.append_byte(byte)
  }
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
  }
}

///|
fn runtime_input_code(input : @plan.OperationRuntimeInput) -> Int {
  match input {
    StepCounts => 1
    QueryPositions => 2
    QueryRowOffsets => 3
    SequenceLengths => 4
    PageTableOffsets => 5
    PageTablePageIndices => 6
  }
}

///|
fn step_stage_code(stage : DeviceStepStage) -> Int {
  match stage {
    Counts => 1
    TokenIds => 2
    TokenPositions => 3
    QueryRowOffsets => 4
    SequenceLengths => 5
    PageOffsets => 6
    PhysicalPageIndices => 7
  }
}

///|
fn BootstrapEncoder::append_role(
  self : BootstrapEncoder,
  role : @launch_contract.AotOperandRole,
) -> Unit raise DeviceBootstrapError {
  match role {
    ExternalInput(TokenIds) => {
      self.append_int(1)
      self.append_int(1)
    }
    WeightInput(reference) => {
      self.append_int(2)
      self.append_int(reference.as_int())
    }
    ActivationInput(reference) => {
      self.append_int(3)
      self.append_int(reference.as_int())
    }
    RuntimeInput(input) => {
      self.append_int(4)
      self.append_int(runtime_input_code(input))
    }
    PersistentKvComponent(layer, component) => {
      self.append_int(5)
      self.append_int(layer.as_int())
      self.append_int(if component == Key { 1 } else { 2 })
    }
    ActivationOutput(reference) => {
      self.append_int(6)
      self.append_int(reference.as_int())
    }
    Workspace => self.append_int(7)
  }
}

///|
fn BootstrapEncoder::append_space(
  self : BootstrapEncoder,
  space : PagedKvOperandSpace,
) -> Unit raise DeviceBootstrapError {
  match space {
    StepDescriptor(stage) => {
      self.append_int(1)
      self.append_int(step_stage_code(stage))
    }
    WeightArena => self.append_int(2)
    ActivationWorkspaceArena => self.append_int(3)
    PersistentKvArena => self.append_int(4)
  }
}

///|
fn BootstrapEncoder::append_entry_identity(
  self : BootstrapEncoder,
  entry : @catalog.AotKernelEntryPoint,
) -> Unit raise DeviceBootstrapError {
  self.append_string(entry.family().module_digest().sha256())
  self.append_int(entry.family().family_id().as_int())
  self.append_int(entry.entry_point_id().as_int())
}

///|
fn BootstrapEncoder::append_dimensions(
  self : BootstrapEncoder,
  dimensions : @launch_contract.AotLaunchDimensions,
) -> Unit raise DeviceBootstrapError {
  self.append_int(dimensions.grid_x())
  self.append_int(dimensions.grid_y())
  self.append_int(dimensions.grid_z())
  self.append_int(dimensions.block_x())
  self.append_int(dimensions.block_y())
  self.append_int(dimensions.block_z())
  self.append_int(dimensions.shared_memory_bytes())
}
