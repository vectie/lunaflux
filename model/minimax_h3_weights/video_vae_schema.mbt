///|
/// Exact released MiniMax H3 video-VAE tensor schema authenticated against the
/// pinned official configuration, safetensors index, and three file headers.
/// This is host metadata authority only; it is not VAE execution authority.
pub struct MiniMaxH3VideoVaeSchema {
  priv version : Int
} derive(Debug, Eq)

///|
pub fn MiniMaxH3VideoVaeSchema::official_v1() -> Self {
  { version: 1, }
}

///|
pub fn MiniMaxH3VideoVaeSchema::version(self : Self) -> Int {
  self.version
}

///|
pub fn MiniMaxH3VideoVaeSchema::source_revision(_self : Self) -> String {
  "42ed227ee7df40d41602854ae760620d6eb651fe"
}

///|
pub fn MiniMaxH3VideoVaeSchema::config_sha256(_self : Self) -> String {
  "78f67deec3d63aae807f2bfe7154bc1e26f6372cb20b63265fcbae1b62bb5745"
}

///|
pub fn MiniMaxH3VideoVaeSchema::index_sha256(_self : Self) -> String {
  "15f6d44553c3c616b0dc999920aa784f92ecee7e4201f1f99ac405cfbf3061ca"
}

///|
pub fn MiniMaxH3VideoVaeSchema::tensor_count(_self : Self) -> Int {
  703
}

///|
pub fn MiniMaxH3VideoVaeSchema::tensor_bytes(_self : Self) -> Int64 {
  10415475936L
}

///|
pub fn MiniMaxH3VideoVaeSchema::shard_count(_self : Self) -> Int {
  3
}

///|
pub fn MiniMaxH3VideoVaeSchema::shard_filename(
  _self : Self,
  index : Int,
) -> String? {
  match index {
    0 => Some("diffusion_pytorch_model-00001-of-00003.safetensors")
    1 => Some("diffusion_pytorch_model-00002-of-00003.safetensors")
    2 => Some("diffusion_pytorch_model-00003-of-00003.safetensors")
    _ => None
  }
}

///|
pub fn MiniMaxH3VideoVaeSchema::shard_file_sha256(
  _self : Self,
  index : Int,
) -> String? {
  match index {
    0 =>
      Some("72f4c6be84ac0674f27398cde991dd9d719762f3952c4921aa66b2ce542f6374")
    1 =>
      Some("2e05e8bc23fa4071043e17fd242be8acd0685e781a43987432b2eae925be4198")
    2 =>
      Some("c05d6ac4b1a33de372799d708531da6320f6a3ce6d1ce6d895e770988e004a39")
    _ => None
  }
}

///|
pub fn MiniMaxH3VideoVaeSchema::shard_file_bytes(
  _self : Self,
  index : Int,
) -> Int64? {
  match index {
    0 => Some(5061033024L)
    1 => Some(4955986528L)
    2 => Some(398539336L)
    _ => None
  }
}

///|
pub fn MiniMaxH3VideoVaeSchema::shard_payload_bytes(
  _self : Self,
  index : Int,
) -> Int64? {
  match index {
    0 => Some(5060997120L)
    1 => Some(4955942592L)
    2 => Some(398536224L)
    _ => None
  }
}

///|
pub fn MiniMaxH3VideoVaeSchema::shard_tensor_count(
  _self : Self,
  index : Int,
) -> Int? {
  match index {
    0 => Some(303)
    1 => Some(372)
    2 => Some(28)
    _ => None
  }
}

///|
pub fn MiniMaxH3VideoVaeSchema::tensor_name(
  self : Self,
  index : Int,
) -> String? {
  if index < 0 || index >= self.tensor_count() {
    return None
  }
  Some(expected_video_vae_names()[index])
}

///|
pub fn MiniMaxH3VideoVaeSchema::tensor_shape(
  self : Self,
  index : Int,
) -> ReadOnlyArray[Int64]? {
  guard self.tensor_name(index) is Some(name) else { return None }
  match video_vae_shape(name) {
    Some(shape) => Some(ReadOnlyArray::from_array(shape))
    None => None
  }
}

///|
pub fn MiniMaxH3VideoVaeSchema::tensor_dtype(
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
pub fn MiniMaxH3VideoVaeSchema::matches_tensor(
  _self : Self,
  name : String,
  dtype : @safetensors.TensorDType,
  shape : ReadOnlyArray[Int64],
) -> Bool {
  video_vae_tensor_is_exact(name, dtype, shape)
}
