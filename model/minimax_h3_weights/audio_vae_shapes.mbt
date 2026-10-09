///|
fn audio_stage_channels(stage : Int) -> (Int64, Int64)? {
  match stage {
    0 => Some((64L, 128L))
    1 => Some((128L, 256L))
    2 => Some((256L, 512L))
    3 => Some((512L, 1024L))
    4 => Some((1024L, 2048L))
    _ => None
  }
}

///|
fn audio_encoder_shape(parts : Array[StringView]) -> Array[Int64]? {
  if parts.length() >= 4 && parts[0] == "encoder" && parts[1] == "block" {
    match (parts[2], parts[3:].to_owned()) {
      ("0", ["bias"]) => return Some([64L])
      ("0", ["weight_g"]) => return Some([64L, 1L, 1L])
      ("0", ["weight_v"]) => return Some([64L, 1L, 7L])
      ("6", ["alpha"]) => return Some([1L, 2048L, 1L])
      ("7", ["bias"]) => return Some([2048L])
      ("7", ["weight_g"]) => return Some([2048L, 1L, 1L])
      ("7", ["weight_v"]) => return Some([2048L, 2048L, 3L])
      _ => ()
    }
  }
  if parts.length() < 6 ||
    parts[0] != "encoder" ||
    parts[1] != "block" ||
    !bounded_index(parts[2], 6) ||
    parts[2] == "0" ||
    parts[3] != "block" {
    return None
  }
  let stage = @string.parse_int(parts[2], base=10) catch { _ => return None }
  guard audio_stage_channels(stage - 1) is Some((input, output)) else {
    return None
  }
  if parts_match(parts, 4, ["3", "alpha"]) {
    return Some([1L, input, 1L])
  }
  if parts_match(parts, 4, ["4", "bias"]) {
    return Some([output])
  }
  if parts_match(parts, 4, ["4", "weight_g"]) {
    return Some([output, 1L, 1L])
  }
  if parts_match(parts, 4, ["4", "weight_v"]) {
    let kernel = match stage {
      1 => 4L
      2 | 3 => 8L
      4 | 5 => 10L
      _ => return None
    }
    return Some([output, input, kernel])
  }
  if parts.length() != 8 || !bounded_index(parts[4], 3) || parts[5] != "block" {
    return None
  }
  match (parts[6], parts[7]) {
    ("0" | "2", "alpha") => Some([1L, input, 1L])
    ("1" | "3", "bias") => Some([input])
    ("1" | "3", "weight_g") => Some([input, 1L, 1L])
    ("1", "weight_v") => Some([input, input, 7L])
    ("3", "weight_v") => Some([input, input, 1L])
    _ => None
  }
}

///|
fn audio_decoder_channels(block : Int) -> Int64? {
  match block / 3 {
    0 => Some(512L)
    1 => Some(256L)
    2 => Some(128L)
    3 => Some(64L)
    4 => Some(32L)
    5 => Some(16L)
    6 => Some(8L)
    _ => None
  }
}

///|
fn audio_decoder_resblock_shape(parts : Array[StringView]) -> Array[Int64]? {
  if parts.length() < 6 ||
    parts[0] != "decoder" ||
    parts[1] != "resblocks" ||
    !bounded_index(parts[2], 21) {
    return None
  }
  let block = @string.parse_int(parts[2], base=10) catch { _ => return None }
  guard audio_decoder_channels(block) is Some(channels) else { return None }
  if parts.length() == 7 &&
    parts[3] == "activations" &&
    bounded_index(parts[4], 6) &&
    (
      parts_match(parts, 5, ["act", "alpha"]) ||
      parts_match(parts, 5, ["act", "beta"])
    ) {
    return Some([channels])
  }
  if parts[3] == "activations" &&
    bounded_index(parts[4], 6) &&
    (
      parts_match(parts, 5, ["downsample", "lowpass", "filter"]) ||
      parts_match(parts, 5, ["upsample", "filter"])
    ) {
    return Some([1L, 1L, 12L])
  }
  if parts.length() != 6 ||
    (parts[3] != "convs1" && parts[3] != "convs2") ||
    !bounded_index(parts[4], 3) {
    return None
  }
  match parts[5] {
    "bias" => Some([channels])
    "weight_g" => Some([channels, 1L, 1L])
    "weight_v" => {
      let kernel = match block % 3 {
        0 => 3L
        1 => 7L
        _ => 11L
      }
      Some([channels, channels, kernel])
    }
    _ => None
  }
}

///|
fn audio_decoder_up_shape(parts : Array[StringView]) -> Array[Int64]? {
  if parts.length() != 5 ||
    parts[0] != "decoder" ||
    parts[1] != "ups" ||
    !bounded_index(parts[2], 7) ||
    parts[3] != "0" {
    return None
  }
  let stage = @string.parse_int(parts[2], base=10) catch { _ => return None }
  let input = 1024L >> stage
  let output = input / 2L
  let kernel = if stage < 2 { 9L } else { 4L }
  match parts[4] {
    "bias" => Some([output])
    "weight_g" => Some([input, 1L, 1L])
    "weight_v" => Some([input, output, kernel])
    _ => None
  }
}

///|
fn audio_pre_block_shape(name : String) -> Array[Int64]? {
  match name {
    "pre_block.attn.proj.bias" => Some([32L])
    "pre_block.attn.proj.weight" => Some([32L, 32L])
    "pre_block.attn.q_bias"
    | "pre_block.attn.v_bias"
    | "pre_block.attn.zero_k_bias" => Some([2048L])
    "pre_block.attn.qkv.weight" => Some([6144L, 2048L])
    "pre_block.mlp.norm.bias" | "pre_block.mlp.norm.weight" => Some([32L])
    "pre_block.mlp.w0.bias" | "pre_block.mlp.w1.bias" => Some([64L])
    "pre_block.mlp.w0.weight" | "pre_block.mlp.w1.weight" => Some([64L, 32L])
    "pre_block.mlp.w2.bias" => Some([32L])
    "pre_block.mlp.w2.weight" => Some([32L, 64L])
    "pre_block.norm1.bias"
    | "pre_block.norm1.weight"
    | "pre_block.norm3.bias"
    | "pre_block.norm3.weight" => Some([2048L])
    "pre_block.norm2.bias" | "pre_block.norm2.weight" => Some([32L])
    "pre_block.proj.bias" => Some([32L])
    "pre_block.proj.weight" => Some([32L, 2048L])
    _ => None
  }
}

///|
fn audio_vae_shape(name : String) -> Array[Int64]? {
  match name {
    "dec_in_proj.weight" => Some([2048L, 32L, 1L])
    "dec_in_proj.bias" => Some([2048L])
    "logs_proj.weight" | "mean_proj.weight" => Some([32L, 32L, 1L])
    "logs_proj.bias" | "mean_proj.bias" => Some([32L])
    "decoder.activation_post.act.alpha" | "decoder.activation_post.act.beta" =>
      Some([8L])
    "decoder.activation_post.downsample.lowpass.filter"
    | "decoder.activation_post.upsample.filter" => Some([1L, 1L, 12L])
    "decoder.conv_post.weight_g" => Some([1L, 1L, 1L])
    "decoder.conv_post.weight_v" => Some([1L, 8L, 7L])
    "decoder.conv_pre.bias" => Some([1024L])
    "decoder.conv_pre.weight_g" => Some([1024L, 1L, 1L])
    "decoder.conv_pre.weight_v" => Some([1024L, 2048L, 7L])
    _ =>
      match audio_pre_block_shape(name) {
        Some(shape) => Some(shape)
        None => {
          let parts = name.split(".").to_array()
          match audio_encoder_shape(parts) {
            Some(shape) => Some(shape)
            None =>
              match audio_decoder_resblock_shape(parts) {
                Some(shape) => Some(shape)
                None => audio_decoder_up_shape(parts)
              }
          }
        }
      }
  }
}

///|
fn audio_vae_tensor_is_exact(
  name : String,
  dtype : @safetensors.TensorDType,
  shape : ReadOnlyArray[Int64],
) -> Bool {
  if dtype != Float32 {
    return false
  }
  match audio_vae_shape(name) {
    Some(expected) => video_vae_shape_is(shape, expected)
    None => false
  }
}
