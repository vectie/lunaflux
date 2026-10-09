///|
pub struct VideoAttentionWeightPacking {
  layer : Int
  byte_count : Int64
  copies : ReadOnlyArray[BlockWeightCopy]
}

///|
pub fn video_attention_weight_packing(
  layer~ : Int,
  max_bytes~ : Int64,
) -> VideoAttentionWeightPacking raise BlockPackingError {
  guard layer >= 0 && layer < 36 else { raise BudgetExceeded }
  let prefix = "decoder.transformer_blocks.\{layer}"
  let names = [
    prefix + ".norm1.weight",
    prefix + ".attn.to_q.weight",
    prefix + ".attn.to_q.bias",
    prefix + ".attn.to_k.weight",
    prefix + ".attn.to_k.bias",
    prefix + ".attn.to_v.weight",
    prefix + ".attn.to_v.bias",
  ]
  let shapes : Array[ReadOnlyArray[Int64]] = [
    [2048L],
    [2048L, 2048L],
    [2048L],
    [2048L, 2048L],
    [2048L],
    [2048L, 2048L],
    [2048L],
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
  { layer, byte_count, copies: ReadOnlyArray::from_array(copies), }
}
