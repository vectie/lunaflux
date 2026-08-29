///|
fn candidate_operation_id(
  candidate : LunaBf16KernelCandidate,
) -> @plan.OperationId {
  match candidate {
    Pointwise(value) => value.operation_id()
    Projection(value) => value.operation_id()
    PagedAttention(value) => value.operation_id()
  }
}

///|
pub fn LunaBf16KernelCandidate::operation_id(
  self : LunaBf16KernelCandidate,
) -> @plan.OperationId {
  candidate_operation_id(self)
}

///|
fn candidate_target(
  candidate : LunaBf16KernelCandidate,
) -> @catalog.DeviceTarget {
  match candidate {
    Pointwise(value) => value.target()
    Projection(value) => value.target()
    PagedAttention(value) => value.target()
  }
}

///|
pub fn LunaBf16KernelCandidate::target(
  self : LunaBf16KernelCandidate,
) -> @catalog.DeviceTarget {
  candidate_target(self)
}

///|
fn candidate_profile(
  candidate : LunaBf16KernelCandidate,
) -> @launch_contract.PagedKernelProfile {
  match candidate {
    Pointwise(value) => value.profile()
    Projection(value) => value.profile()
    PagedAttention(value) => value.profile()
  }
}

///|
fn candidate_entry_point_id(
  candidate : LunaBf16KernelCandidate,
) -> @catalog.AotKernelEntryPointId {
  match candidate {
    Pointwise(value) => value.entry_point_id()
    Projection(value) => value.entry_point_id()
    PagedAttention(value) => value.entry_point_id()
  }
}

///|
fn candidate_dimensions(
  candidate : LunaBf16KernelCandidate,
) -> @launch_contract.AotLaunchDimensions {
  match candidate {
    Pointwise(value) => value.launch_dimensions()
    Projection(value) => value.launch_dimensions()
    PagedAttention(value) => value.launch_dimensions()
  }
}

///|
fn candidate_operands(
  candidate : LunaBf16KernelCandidate,
) -> ReadOnlyArray[@launch_contract.AotOperand] {
  match candidate {
    Pointwise(value) => value.operands()
    Projection(value) => value.operands()
    PagedAttention(value) => value.operands()
  }
}

///|
fn candidate_function_symbol(candidate : LunaBf16KernelCandidate) -> String {
  match candidate {
    Pointwise(value) => value.function_symbol()
    Projection(value) => value.function_symbol()
    PagedAttention(value) => value.function_symbol()
  }
}

///|
fn candidate_source_digest(candidate : LunaBf16KernelCandidate) -> String {
  match candidate {
    Pointwise(value) => value.source_digest().as_hex()
    Projection(value) => value.source_digest().as_hex()
    PagedAttention(value) => value.source_digest().as_hex()
  }
}

///|
pub fn LunaBf16KernelCandidate::source_sha256(
  self : LunaBf16KernelCandidate,
) -> String {
  candidate_source_digest(self)
}

///|
pub fn LunaBf16KernelCandidate::recipe_sha256(
  self : LunaBf16KernelCandidate,
) -> String {
  candidate_recipe_digest(self)
}

///|
pub fn LunaBf16KernelCandidate::family_name(
  self : LunaBf16KernelCandidate,
) -> String {
  candidate_family_name(self)
}

///|
fn candidate_kind_matches(
  candidate : LunaBf16KernelCandidate,
  kind : @plan.OperationKind,
) -> Bool {
  match candidate {
    Pointwise(value) =>
      match (value.family(), kind) {
        (EmbeddingLookup, Embedding)
        | (RmsNorm, RmsNorm)
        | (PositionedRotary, RotaryEmbedding)
        | (ResidualAdd, ResidualAdd) => true
        _ => false
      }
    Projection(value) =>
      match (value.family(), kind) {
        (QkvProjection, QkvProjection)
        | (DenseProjection, OutputProjection)
        | (GatedMlp, GatedMlp)
        | (LanguageModelHead, LanguageModelHead) => true
        _ => false
      }
    PagedAttention(_) => kind == CausalAttention
  }
}

///|
fn sha256_bytes(bytes : Bytes) -> String {
  @crypto.bytes_to_hex_string(@crypto.sha256(bytes))
}
