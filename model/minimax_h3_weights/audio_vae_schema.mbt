///|
/// Exact released MiniMax H3 AudioVae schema. The release is one safetensors
/// file without a separate index; its authenticated header is the vocabulary
/// authority. This type grants metadata authority only, never execution.
pub struct MiniMaxH3AudioVaeSchema {
  priv version : Int
} derive(Debug, Eq)

///|
pub fn MiniMaxH3AudioVaeSchema::official_v1() -> Self {
  { version: 1, }
}

///|
pub fn MiniMaxH3AudioVaeSchema::version(self : Self) -> Int {
  self.version
}

///|
pub fn MiniMaxH3AudioVaeSchema::source_revision(_self : Self) -> String {
  "42ed227ee7df40d41602854ae760620d6eb651fe"
}

///|
pub fn MiniMaxH3AudioVaeSchema::config_sha256(_self : Self) -> String {
  "9a3c645ff892b376c6f5f4c8685964cd75474731af594ff058492a0000caabb6"
}

///|
pub fn MiniMaxH3AudioVaeSchema::header_sha256(_self : Self) -> String {
  "91eef2ba9c557abb3db1c336b454813e3feab25eb2a7a524599b44ca6ee7bc78"
}

///|
pub fn MiniMaxH3AudioVaeSchema::tensor_count(_self : Self) -> Int {
  1087
}

///|
pub fn MiniMaxH3AudioVaeSchema::tensor_bytes(_self : Self) -> Int64 {
  605306340L
}

///|
pub fn MiniMaxH3AudioVaeSchema::filename(_self : Self) -> String {
  "diffusion_pytorch_model.safetensors"
}

///|
pub fn MiniMaxH3AudioVaeSchema::file_sha256(_self : Self) -> String {
  "52c59e67ba8de5477c81bfbced0327aabf500f1bfdeefd5ee754529241cb26cb"
}

///|
pub fn MiniMaxH3AudioVaeSchema::file_bytes(_self : Self) -> Int64 {
  605429340L
}

///|
pub fn MiniMaxH3AudioVaeSchema::header_bytes(_self : Self) -> Int {
  122992
}

///|
pub fn MiniMaxH3AudioVaeSchema::tensor_name(
  self : Self,
  index : Int,
) -> String? {
  if index < 0 || index >= self.tensor_count() {
    None
  } else {
    Some(expected_audio_vae_names()[index])
  }
}

///|
pub fn MiniMaxH3AudioVaeSchema::tensor_shape(
  self : Self,
  index : Int,
) -> ReadOnlyArray[Int64]? {
  guard self.tensor_name(index) is Some(name) else { return None }
  match audio_vae_shape(name) {
    Some(shape) => Some(ReadOnlyArray::from_array(shape))
    None => None
  }
}

///|
pub fn MiniMaxH3AudioVaeSchema::tensor_dtype(
  self : Self,
  index : Int,
) -> @safetensors.TensorDType? {
  if index < 0 || index >= self.tensor_count() {
    None
  } else {
    Some(Float32)
  }
}

///|
pub fn MiniMaxH3AudioVaeSchema::matches_tensor(
  _self : Self,
  name : String,
  dtype : @safetensors.TensorDType,
  shape : ReadOnlyArray[Int64],
) -> Bool {
  audio_vae_tensor_is_exact(name, dtype, shape)
}
