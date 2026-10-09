///|
fn video_vae_shape_is(
  actual : ReadOnlyArray[Int64],
  expected : ArrayView[Int64],
) -> Bool {
  actual == ReadOnlyArray::from_array(expected)
}

///|
fn video_vae_channels(block : Int) -> (Int64, Int64)? {
  match block {
    0 => Some((128L, 128L))
    1 => Some((128L, 256L))
    2 => Some((256L, 256L))
    3 => Some((256L, 512L))
    4 => Some((512L, 512L))
    5 => Some((512L, 1024L))
    _ => None
  }
}

///|
fn video_vae_decoder_block_shape(parts : Array[StringView]) -> Array[Int64]? {
  if parts.length() < 4 ||
    parts[0] != "decoder" ||
    parts[1] != "transformer_blocks" ||
    !bounded_index(parts[2], 36) {
    return None
  }
  if parts_match(parts, 3, ["attn", "to_k", "weight"]) ||
    parts_match(parts, 3, ["attn", "to_q", "weight"]) ||
    parts_match(parts, 3, ["attn", "to_v", "weight"]) ||
    parts_match(parts, 3, ["attn", "to_out", "0", "weight"]) {
    Some([2048L, 2048L])
  } else if parts_match(parts, 3, ["attn", "to_k", "bias"]) ||
    parts_match(parts, 3, ["attn", "to_q", "bias"]) ||
    parts_match(parts, 3, ["attn", "to_v", "bias"]) ||
    parts_match(parts, 3, ["attn", "to_out", "0", "bias"]) {
    Some([2048L])
  } else if parts_match(parts, 3, ["ff", "net", "0", "proj", "weight"]) {
    Some([16384L, 2048L])
  } else if parts_match(parts, 3, ["ff", "net", "0", "proj", "bias"]) {
    Some([16384L])
  } else if parts_match(parts, 3, ["ff", "net", "2", "weight"]) {
    Some([2048L, 8192L])
  } else if parts_match(parts, 3, ["ff", "net", "2", "bias"]) ||
    parts_match(parts, 3, ["norm1", "weight"]) ||
    parts_match(parts, 3, ["norm2", "weight"]) ||
    parts_match(parts, 3, ["scale1"]) ||
    parts_match(parts, 3, ["scale2"]) {
    Some([2048L])
  } else {
    None
  }
}

///|
fn video_vae_resnet_shape(parts : Array[StringView]) -> Array[Int64]? {
  if parts.length() < 7 ||
    parts[0] != "encoder" ||
    parts[1] != "down_blocks" ||
    !bounded_index(parts[2], 6) ||
    parts[3] != "resnets" ||
    !bounded_index(parts[4], 2) {
    return None
  }
  let block = @string.parse_int(parts[2], base=10) catch { _ => return None }
  let resnet = @string.parse_int(parts[4], base=10) catch { _ => return None }
  guard video_vae_channels(block) is Some((block_input, block_output)) else {
    return None
  }
  let input = if resnet == 0 { block_input } else { block_output }
  if parts_match(parts, 5, ["conv1", "weight"]) {
    Some([block_output, input, 3L, 3L, 3L])
  } else if parts_match(parts, 5, ["conv2", "weight"]) {
    Some([block_output, block_output, 3L, 3L, 3L])
  } else if parts_match(parts, 5, ["conv1", "bias"]) ||
    parts_match(parts, 5, ["conv2", "bias"]) {
    Some([block_output])
  } else if parts_match(parts, 5, ["norm1", "weight"]) ||
    parts_match(parts, 5, ["norm1", "bias"]) {
    Some([input])
  } else if parts_match(parts, 5, ["norm2", "weight"]) ||
    parts_match(parts, 5, ["norm2", "bias"]) {
    Some([block_output])
  } else if resnet == 0 &&
    block_input != block_output &&
    parts_match(parts, 5, ["conv_shortcut", "weight"]) {
    Some([block_output, block_input, 1L, 1L, 1L])
  } else if resnet == 0 &&
    block_input != block_output &&
    parts_match(parts, 5, ["conv_shortcut", "bias"]) {
    Some([block_output])
  } else {
    None
  }
}

///|
fn video_vae_downsample_shape(parts : Array[StringView]) -> Array[Int64]? {
  if parts.length() != 7 ||
    parts[0] != "encoder" ||
    parts[1] != "down_blocks" ||
    !bounded_index(parts[2], 4) ||
    parts[3] != "downsamplers" ||
    parts[4] != "0" ||
    parts[5] != "conv" {
    return None
  }
  let block = @string.parse_int(parts[2], base=10) catch { _ => return None }
  guard video_vae_channels(block) is Some((_, channels)) else { return None }
  if parts[6] == "weight" {
    Some([channels, channels, 3L, 3L, 3L])
  } else if parts[6] == "bias" {
    Some([channels])
  } else {
    None
  }
}

///|
fn video_vae_shape(name : String) -> Array[Int64]? {
  match name {
    "decoder.norm_out.weight" | "decoder.norm_out.bias" => Some([2048L])
    "decoder.proj_in.weight" => Some([2048L, 24L])
    "decoder.proj_in.bias" => Some([2048L])
    "decoder.proj_out.weight" => Some([3072L, 2048L])
    "decoder.proj_out.bias" => Some([3072L])
    "decoder.register_tokens" => Some([1L, 4L, 2048L])
    "encoder.conv_in.weight" => Some([128L, 3L, 3L, 3L, 3L])
    "encoder.conv_in.bias" => Some([128L])
    "encoder.conv_out.weight" => Some([48L, 1024L, 3L, 3L, 3L])
    "encoder.conv_out.bias" => Some([48L])
    "encoder.norm_out.weight" | "encoder.norm_out.bias" => Some([1024L])
    "post_quant_conv.weight" => Some([24L, 24L, 1L, 1L, 1L])
    "post_quant_conv.bias" => Some([24L])
    "quant_conv.weight" => Some([48L, 48L, 1L, 1L, 1L])
    "quant_conv.bias" => Some([48L])
    _ => {
      let parts = name.split(".").to_array()
      match video_vae_decoder_block_shape(parts) {
        Some(shape) => Some(shape)
        None =>
          match video_vae_resnet_shape(parts) {
            Some(shape) => Some(shape)
            None => video_vae_downsample_shape(parts)
          }
      }
    }
  }
}

///|
fn video_vae_tensor_is_exact(
  name : String,
  dtype : @safetensors.TensorDType,
  shape : ReadOnlyArray[Int64],
) -> Bool {
  if dtype != Float32 {
    return false
  }
  match video_vae_shape(name) {
    Some(expected) => video_vae_shape_is(shape, expected)
    None => false
  }
}
