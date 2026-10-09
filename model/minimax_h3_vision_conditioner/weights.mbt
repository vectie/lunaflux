///|
pub struct WeightRegion {
  name : String
  offset : Int64
  bytes : Int64
  shape : ReadOnlyArray[Int64]
}

///|
pub struct WeightGroup {
  key : String
  regions : ReadOnlyArray[WeightRegion]
  bytes : Int64
}

///|
fn weight_group(
  key : String,
  prefix : String,
  tensors : ArrayView[(String, Array[Int64])],
) -> WeightGroup {
  let regions = []
  let mut offset = 0L
  for (suffix, shape) in tensors {
    let elements = shape.fold(init=1L, (a, b) => a * b)
    let bytes = elements * 2L
    regions.push({
      name: prefix + "." + suffix,
      offset,
      bytes,
      shape: ReadOnlyArray::from_array(shape),
    })
    offset += bytes
  }
  { key, regions: ReadOnlyArray::from_array(regions), bytes: offset, }
}

///|
pub fn frontend_weights() -> WeightGroup {
  weight_group("frontend", "model.visual", [
    ("patch_embed.proj.weight", [1152L, 3L, 2L, 16L, 16L]),
    ("patch_embed.proj.bias", [1152L]),
    ("pos_embed.weight", [2304L, 1152L]),
  ])
}

///|
pub fn block_weights(layer : Int) -> WeightGroup raise VisionError {
  guard layer >= 0 && layer < 27 else { raise InvalidGeometry }
  weight_group("block\{layer}", "model.visual.blocks.\{layer}", [
    ("norm1.weight", [1152L]),
    ("norm1.bias", [1152L]),
    ("attn.qkv.weight", [3456L, 1152L]),
    ("attn.qkv.bias", [3456L]),
    ("attn.proj.weight", [1152L, 1152L]),
    ("attn.proj.bias", [1152L]),
    ("norm2.weight", [1152L]),
    ("norm2.bias", [1152L]),
    ("mlp.linear_fc1.weight", [4304L, 1152L]),
    ("mlp.linear_fc1.bias", [4304L]),
    ("mlp.linear_fc2.weight", [1152L, 4304L]),
    ("mlp.linear_fc2.bias", [1152L]),
  ])
}

///|
pub fn merger_weights(index : Int) -> WeightGroup raise VisionError {
  guard index >= 0 && index < 4 else { raise InvalidGeometry }
  let prefix = if index == 0 {
    "model.visual.merger"
  } else {
    "model.visual.deepstack_merger_list.\{index - 1}"
  }
  let norm = if index == 0 { 1152L } else { 4608L }
  weight_group("merger\{index}", prefix, [
    ("norm.weight", [norm]),
    ("norm.bias", [norm]),
    ("linear_fc1.weight", [4608L, 4608L]),
    ("linear_fc1.bias", [4608L]),
    ("linear_fc2.weight", [5120L, 4608L]),
    ("linear_fc2.bias", [5120L]),
  ])
}
