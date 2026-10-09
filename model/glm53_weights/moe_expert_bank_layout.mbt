///|
pub(all) suberror Glm53ExpertBankLayoutError {
  UnsupportedExpertBankEncoding
  MissingExpertTensor(String)
  ExpertTensorMismatch(String)
  ExpertBankSizeOverflow
  IndexedLayout(@indexed_parameter_bank_layout.IndexedParameterBankLayoutError)
} derive(Debug, Eq)

///|
pub struct Glm53MoeExpertBankLayout {
  priv execution_identity : @spec.ModelIdentity
  priv manifest_layout_digest : @spec.PlanDigest
  priv layer_index : Int
  priv routed_expert_count : Int
  priv hidden_width : Int
  priv intermediate_width : Int
  priv routed_gate : @indexed_parameter_bank_layout.IndexedParameterBankLayout
  priv routed_up : @indexed_parameter_bank_layout.IndexedParameterBankLayout
  priv routed_down : @indexed_parameter_bank_layout.IndexedParameterBankLayout
  priv shared_gate : @indexed_parameter_bank_layout.IndexedParameterBankLayout
  priv shared_up : @indexed_parameter_bank_layout.IndexedParameterBankLayout
  priv shared_down : @indexed_parameter_bank_layout.IndexedParameterBankLayout
  priv digest : Glm53MoeExpertBankLayoutDigest
}

///|
pub struct Glm53MoeExpertBankLayoutDigest(String) derive(Debug, Eq, Hash)

///|
fn find_expert_tensor(
  manifest : Glm53TensorManifest,
  name : String,
  shape : ArrayView[Int64],
) -> Glm53TensorDescriptor raise Glm53ExpertBankLayoutError {
  let mut found : Glm53TensorDescriptor? = None
  for ordinal in 0..<manifest.tensor_count() {
    guard manifest.tensor(ordinal) is Some(tensor) else { continue }
    if tensor.name() == name {
      if found is Some(_) {
        raise ExpertTensorMismatch(name)
      }
      found = Some(tensor)
    }
  }
  guard found is Some(tensor) else { raise MissingExpertTensor(name) }
  let mut shape_matches = tensor.shape().length() == shape.length()
  if shape_matches {
    for index, dimension in shape {
      if tensor.shape()[index] != dimension {
        shape_matches = false
      }
    }
  }
  if tensor.dtype() != BFloat16 || !shape_matches {
    raise ExpertTensorMismatch(name)
  }
  tensor
}

///|
fn expert_tensor_bytes(
  hidden_width : Int,
  intermediate_width : Int,
) -> Int64 raise Glm53ExpertBankLayoutError {
  let hidden = hidden_width.to_int64()
  let intermediate = intermediate_width.to_int64()
  if hidden <= 0L ||
    intermediate <= 0L ||
    hidden > 9223372036854775807L / intermediate ||
    hidden * intermediate > 9223372036854775807L / 2L {
    raise ExpertBankSizeOverflow
  }
  hidden * intermediate * 2L
}

///|
fn build_expert_bank(
  manifest : Glm53TensorManifest,
  prefix : String,
  suffix : String,
  expert_count : Int,
  shape : ArrayView[Int64],
  tensor_bytes : Int64,
) -> @indexed_parameter_bank_layout.IndexedParameterBankLayout raise Glm53ExpertBankLayoutError {
  let inputs : Array[@indexed_parameter_bank_layout.IndexedParameterSliceInput] = []
  for expert in 0..<expert_count {
    let name = prefix + expert.to_string() + suffix
    let tensor = find_expert_tensor(manifest, name, shape)
    inputs.push(
      @indexed_parameter_bank_layout.IndexedParameterSliceInput::new(
        logical_index=expert,
        source_ordinal=tensor.ordinal(),
        source_name=tensor.name(),
        byte_count=tensor_bytes,
      ) catch {
        error => raise IndexedLayout(error)
      },
    )
  }
  let count = expert_count.to_int64()
  if count <= 0L || count > 9223372036854775807L / tensor_bytes {
    raise ExpertBankSizeOverflow
  }
  @indexed_parameter_bank_layout.build_indexed_parameter_bank_layout(
    inputs,
    alignment=16L,
    maximum_byte_count=count * tensor_bytes,
  ) catch {
    error => raise IndexedLayout(error)
  }
}

///|
fn build_shared_bank(
  manifest : Glm53TensorManifest,
  name : String,
  shape : ArrayView[Int64],
  tensor_bytes : Int64,
) -> @indexed_parameter_bank_layout.IndexedParameterBankLayout raise Glm53ExpertBankLayoutError {
  let tensor = find_expert_tensor(manifest, name, shape)
  let input = @indexed_parameter_bank_layout.IndexedParameterSliceInput::new(
    logical_index=0,
    source_ordinal=tensor.ordinal(),
    source_name=tensor.name(),
    byte_count=tensor_bytes,
  ) catch {
    error => raise IndexedLayout(error)
  }
  @indexed_parameter_bank_layout.build_indexed_parameter_bank_layout(
    [input],
    alignment=16L,
    maximum_byte_count=tensor_bytes,
  ) catch {
    error => raise IndexedLayout(error)
  }
}

///|
/// Authenticates the startup-only projection from per-expert checkpoint tensors
/// into the expert-major banks consumed by the correctness CUDA candidate.
/// It performs no allocation, copy, upload, or launch.
pub fn build_moe_expert_bank_layout(
  manifest : Glm53TensorManifest,
  layer_index~ : Int,
) -> Glm53MoeExpertBankLayout raise Glm53ExpertBankLayoutError {
  if manifest.encoding() != BFloat16WithFloat32Routing {
    raise UnsupportedExpertBankEncoding
  }
  let (prefix, expert_count, hidden_width) = match manifest.profile() {
    FullText => ("model.layers.", 256, 6144)
    FlashMultimodal => ("model.language_model.layers.", 288, 4096)
  }
  let intermediate_width = 2048
  let layer_prefix = prefix + layer_index.to_string() + ".mlp."
  let expert_prefix = layer_prefix + "experts."
  let tensor_bytes = expert_tensor_bytes(hidden_width, intermediate_width)
  let gate_shape = [intermediate_width.to_int64(), hidden_width.to_int64()]
  let down_shape = [hidden_width.to_int64(), intermediate_width.to_int64()]
  let routed_gate = build_expert_bank(
    manifest, expert_prefix, ".gate_proj.weight", expert_count, gate_shape, tensor_bytes,
  )
  let routed_up = build_expert_bank(
    manifest, expert_prefix, ".up_proj.weight", expert_count, gate_shape, tensor_bytes,
  )
  let routed_down = build_expert_bank(
    manifest, expert_prefix, ".down_proj.weight", expert_count, down_shape, tensor_bytes,
  )
  let shared_prefix = layer_prefix + "shared_experts."
  let shared_gate = build_shared_bank(
    manifest,
    shared_prefix + "gate_proj.weight",
    gate_shape,
    tensor_bytes,
  )
  let shared_up = build_shared_bank(
    manifest,
    shared_prefix + "up_proj.weight",
    gate_shape,
    tensor_bytes,
  )
  let shared_down = build_shared_bank(
    manifest,
    shared_prefix + "down_proj.weight",
    down_shape,
    tensor_bytes,
  )
  let digest_text = "lunaflux.glm53-moe-expert-bank-layout.v1|" +
    manifest.execution_identity().content().as_hex() +
    "|" +
    manifest.execution_identity().plan().as_hex() +
    "|" +
    manifest.layout_digest().as_hex() +
    "|" +
    layer_index.to_string() +
    "|" +
    expert_count.to_string() +
    ":" +
    hidden_width.to_string() +
    ":" +
    intermediate_width.to_string() +
    "|" +
    routed_gate.digest().as_hex() +
    "|" +
    routed_up.digest().as_hex() +
    "|" +
    routed_down.digest().as_hex() +
    "|" +
    shared_gate.digest().as_hex() +
    "|" +
    shared_up.digest().as_hex() +
    "|" +
    shared_down.digest().as_hex()
  {
    execution_identity: manifest.execution_identity(),
    manifest_layout_digest: manifest.layout_digest(),
    layer_index,
    routed_expert_count: expert_count,
    hidden_width,
    intermediate_width,
    routed_gate,
    routed_up,
    routed_down,
    shared_gate,
    shared_up,
    shared_down,
    digest: Glm53MoeExpertBankLayoutDigest(
      @crypto.bytes_to_hex_string(@crypto.sha256(@utf8.encode(digest_text))),
    ),
  }
}

///|
pub fn Glm53MoeExpertBankLayout::execution_identity(
  self : Self,
) -> @spec.ModelIdentity {
  self.execution_identity
}

///|
pub fn Glm53MoeExpertBankLayout::manifest_layout_digest(
  self : Self,
) -> @spec.PlanDigest {
  self.manifest_layout_digest
}

///|
pub fn Glm53MoeExpertBankLayout::layer_index(self : Self) -> Int {
  self.layer_index
}

///|
pub fn Glm53MoeExpertBankLayout::routed_expert_count(self : Self) -> Int {
  self.routed_expert_count
}

///|
pub fn Glm53MoeExpertBankLayout::hidden_width(self : Self) -> Int {
  self.hidden_width
}

///|
pub fn Glm53MoeExpertBankLayout::intermediate_width(self : Self) -> Int {
  self.intermediate_width
}

///|
pub fn Glm53MoeExpertBankLayout::routed_gate(
  self : Self,
) -> @indexed_parameter_bank_layout.IndexedParameterBankLayout {
  self.routed_gate
}

///|
pub fn Glm53MoeExpertBankLayout::routed_up(
  self : Self,
) -> @indexed_parameter_bank_layout.IndexedParameterBankLayout {
  self.routed_up
}

///|
pub fn Glm53MoeExpertBankLayout::routed_down(
  self : Self,
) -> @indexed_parameter_bank_layout.IndexedParameterBankLayout {
  self.routed_down
}

///|
pub fn Glm53MoeExpertBankLayout::shared_gate(
  self : Self,
) -> @indexed_parameter_bank_layout.IndexedParameterBankLayout {
  self.shared_gate
}

///|
pub fn Glm53MoeExpertBankLayout::shared_up(
  self : Self,
) -> @indexed_parameter_bank_layout.IndexedParameterBankLayout {
  self.shared_up
}

///|
pub fn Glm53MoeExpertBankLayout::shared_down(
  self : Self,
) -> @indexed_parameter_bank_layout.IndexedParameterBankLayout {
  self.shared_down
}

///|
pub fn Glm53MoeExpertBankLayout::digest(
  self : Self,
) -> Glm53MoeExpertBankLayoutDigest {
  self.digest
}

///|
pub fn Glm53MoeExpertBankLayoutDigest::as_hex(self : Self) -> String {
  self.0
}

///|
pub fn Glm53MoeExpertBankLayout::materialization_authority(
  _self : Self,
) -> Bool {
  false
}
