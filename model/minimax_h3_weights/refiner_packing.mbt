///|
pub struct RefinerWeightPacking {
  layer : Int
  attention_offset : Int64
  mlp_offset : Int64
  norm_offset : Int64
  final_norm_offset : Int64?
  byte_count : Int64
  copies : ReadOnlyArray[BlockWeightCopy]
}

///|
pub fn refiner_weight_packing(
  model : @joint_diffusion_plan.JointDiffusionPlan,
  layer~ : Int,
  max_bytes~ : Int64,
) -> RefinerWeightPacking raise BlockPackingError {
  guard layer >= 0 && layer < model.denoiser().refiner_layer_count() else {
    raise InvalidBlock
  }
  let p = pack_transformer(
    model,
    layer,
    "token_refiner.refiner_blocks.\{layer}.",
    max_bytes,
    false,
  )
  let copies = p.copies[:].to_owned()
  let final_bytes = if layer + 1 == model.denoiser().refiner_layer_count() {
    model.denoiser().hidden_size().to_int64() * 2L
  } else {
    0L
  }
  guard final_bytes <= max_bytes - p.byte_count else { raise BudgetExceeded }
  let final_norm_offset = if final_bytes > 0L {
    copies.push({
      tensor_name: "token_refiner.final_norm.weight",
      source_offset: 0L,
      destination_offset: p.byte_count,
      byte_count: final_bytes,
    })
    Some(p.byte_count)
  } else {
    None
  }
  {
    layer,
    attention_offset: p.attention_offset,
    mlp_offset: p.mlp_offset,
    norm_offset: p.norm_offset,
    final_norm_offset,
    byte_count: p.byte_count + final_bytes,
    copies: ReadOnlyArray::from_array(copies),
  }
}
