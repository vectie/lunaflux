///|
pub struct VideoFrontendWeightPacking {
  byte_count : Int64
  copies : ReadOnlyArray[BlockWeightCopy]
}

///|
/// Official F32 post-quant 1x1x1 convolution, embedding, and four register rows.
pub fn video_frontend_weight_packing(
  max_bytes~ : Int64,
) -> VideoFrontendWeightPacking raise BlockPackingError {
  let names = [
    "post_quant_conv.weight", "post_quant_conv.bias", "decoder.proj_in.weight", "decoder.proj_in.bias",
    "decoder.register_tokens",
  ]
  let shapes : Array[ReadOnlyArray[Int64]] = [
    [24L, 24L, 1L, 1L, 1L],
    [24L],
    [2048L, 24L],
    [2048L],
    [1L, 4L, 2048L],
  ]
  let schema = MiniMaxH3VideoVaeSchema::official_v1()
  let copies : Array[BlockWeightCopy] = []
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
