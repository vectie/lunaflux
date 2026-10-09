///|
pub struct VideoBlockWeightPacking {
  layer : Int
  byte_count : Int64
  copies : ReadOnlyArray[BlockWeightCopy]
}

///|
pub fn video_block_weight_packing(
  layer~ : Int,
  max_bytes~ : Int64,
) -> VideoBlockWeightPacking raise BlockPackingError {
  let stem = video_attention_weight_packing(layer~, max_bytes~)
  let copies = stem.copies[:].to_owned()
  let p = "decoder.transformer_blocks.\{layer}"
  let names = [
    p + ".attn.to_out.0.weight",
    p + ".attn.to_out.0.bias",
    p + ".scale1",
    p + ".norm2.weight",
    p + ".ff.net.0.proj.weight",
    p + ".ff.net.0.proj.bias",
    p + ".ff.net.2.weight",
    p + ".ff.net.2.bias",
    p + ".scale2",
  ]
  let shapes : Array[ReadOnlyArray[Int64]] = [
    [2048L, 2048L],
    [2048L],
    [2048L],
    [2048L],
    [16384L, 2048L],
    [16384L],
    [2048L, 8192L],
    [2048L],
    [2048L],
  ]
  let schema = MiniMaxH3VideoVaeSchema::official_v1()
  let byte_count = for i, name in names; offset = stem.byte_count {
    guard schema.matches_tensor(name, Float32, shapes[i]) else {
      raise BudgetExceeded
    }
    let size = shapes[i].fold(init=4L, (size, extent) => size * extent)
    guard size <= max_bytes - offset else { raise BudgetExceeded }
    copies.push({
      tensor_name: name,
      source_offset: 0L,
      destination_offset: offset,
      byte_count: size,
    })
    continue offset + size
  } nobreak {
    offset
  }
  { layer, byte_count, copies: ReadOnlyArray::from_array(copies), }
}

///|
pub fn video_head_weight_packing(
  max_bytes~ : Int64,
) -> VideoFrontendWeightPacking raise BlockPackingError {
  let names = [
    "decoder.norm_out.weight", "decoder.norm_out.bias", "decoder.proj_out.weight",
    "decoder.proj_out.bias",
  ]
  let shapes : Array[ReadOnlyArray[Int64]] = [
    [2048L],
    [2048L],
    [3072L, 2048L],
    [3072L],
  ]
  let copies : Array[BlockWeightCopy] = []
  let schema = MiniMaxH3VideoVaeSchema::official_v1()
  let byte_count = for i, name in names; offset = 0L {
    guard schema.matches_tensor(name, Float32, shapes[i]) else {
      raise BudgetExceeded
    }
    let size = shapes[i].fold(init=4L, (size, extent) => size * extent)
    guard size <= max_bytes - offset else { raise BudgetExceeded }
    copies.push({
      tensor_name: name,
      source_offset: 0L,
      destination_offset: offset,
      byte_count: size,
    })
    continue offset + size
  } nobreak {
    offset
  }
  { byte_count, copies: ReadOnlyArray::from_array(copies), }
}
