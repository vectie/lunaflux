///|
pub struct BlockWeightCopy {
  tensor_name : String
  source_offset : Int64
  destination_offset : Int64
  byte_count : Int64
}

///|
pub struct BlockWeightPacking {
  layer : Int
  attention_offset : Int64
  mlp_offset : Int64
  norm_offset : Int64
  adaln_offset : Int64
  adaln_bias_offset : Int64
  byte_count : Int64
  copies : ReadOnlyArray[BlockWeightCopy]
}

///|
pub(all) suberror BlockPackingError {
  InvalidBlock
  BudgetExceeded
}

///|
/// Pure copy plan for an admitted H3 denoiser block. Source metadata and model
/// identity must already match the plan. Copies preserve BF16 bytes, including
/// swapping checkpoint [gate,up] halves into the staged runtime [up,gate] ABI.
/// Includes the block's three-modality, six-field AdaLN weight and bias.
pub fn block_weight_packing(
  model : @joint_diffusion_plan.JointDiffusionPlan,
  block~ : Int,
  max_bytes~ : Int64,
) -> BlockWeightPacking raise BlockPackingError {
  guard block >= 0 && block < model.denoiser().layer_count() else {
    raise InvalidBlock
  }
  pack_transformer(
    model,
    block,
    "transformer_blocks.\{block}.",
    max_bytes,
    true,
  )
}

///|
fn pack_transformer(
  model : @joint_diffusion_plan.JointDiffusionPlan,
  block : Int,
  prefix : String,
  max_bytes : Int64,
  with_adaln : Bool,
) -> BlockWeightPacking raise BlockPackingError {
  let shape = model.denoiser()
  let hidden = shape.hidden_size().to_int64()
  let inner = shape.attention_width().to_int64()
  let intermediate = shape.feed_forward_dimension().to_int64()
  // Check before multiplication: the full arena contains four attention
  // matrices, three MLP matrices and four normalization vectors.
  guard max_bytes > 0L &&
    inner <= max_bytes / 8L / hidden &&
    intermediate <= max_bytes / 6L / hidden else {
    raise BudgetExceeded
  }
  let matrix = inner * hidden * 2L
  let mlp_matrix = intermediate * hidden * 2L
  let head_norm = shape.attention_head_dimension().to_int64() * 2L
  let hidden_norm = hidden * 2L
  let copies : Array[BlockWeightCopy] = []
  fn add(
    suffix : String,
    source : Int64,
    size : Int64,
    offset : Int64,
  ) -> Int64 raise BlockPackingError {
    guard size <= max_bytes - offset else { raise BudgetExceeded }
    copies.push({
      tensor_name: prefix + suffix,
      source_offset: source,
      destination_offset: offset,
      byte_count: size,
    })
    offset + size
  }
  let q = add("attn.to_q.weight", 0L, matrix, 0L)
  let k = add("attn.to_k.weight", 0L, matrix, q)
  let v = add("attn.to_v.weight", 0L, matrix, k)
  let qn = add("attn.norm_q.weight", 0L, head_norm, v)
  let kn = add("attn.norm_k.weight", 0L, head_norm, qn)
  let mlp_offset = add("attn.to_out.0.weight", 0L, matrix, kn)
  let up = add("ff.net.0.proj.weight", mlp_matrix, mlp_matrix, mlp_offset)
  let gate = add("ff.net.0.proj.weight", 0L, mlp_matrix, up)
  let norm_offset = add("ff.net.2.weight", 0L, mlp_matrix, gate)
  let norm1 = add("norm1.weight", 0L, hidden_norm, norm_offset)
  let adaln_offset = add("norm2.weight", 0L, hidden_norm, norm1)
  let (adaln_bias_offset, byte_count) = if with_adaln {
    let conditioning = shape.final_adaln_conditioning_dimension().to_int64()
    guard hidden <= max_bytes / 36L / conditioning else { raise BudgetExceeded }
    let adaln_bias_offset = add(
      "adaln_proj.linear.weight",
      0L,
      hidden * 36L * conditioning,
      adaln_offset,
    )
    let byte_count = add(
      "adaln_proj.linear.bias",
      0L,
      hidden * 36L,
      adaln_bias_offset,
    )
    (adaln_bias_offset, byte_count)
  } else {
    (adaln_offset, adaln_offset)
  }
  {
    layer: block,
    attention_offset: 0L,
    mlp_offset,
    norm_offset,
    adaln_offset,
    adaln_bias_offset,
    byte_count,
    copies: ReadOnlyArray::from_array(copies),
  }
}
