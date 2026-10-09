///|
pub struct ContextWeightPacking {
  weight_offset : Int64
  bias_offset : Int64
  byte_count : Int64
  copies : ReadOnlyArray[BlockWeightCopy]
}

///|
/// Pack the checkpoint's BF16 text-context projection without conversion.
pub fn context_weight_packing(
  model : @joint_diffusion_plan.JointDiffusionPlan,
  max_bytes~ : Int64,
) -> ContextWeightPacking raise BlockPackingError {
  let shape = model.denoiser()
  let input = shape.text_conditioning_dimension().to_int64()
  let output = shape.hidden_size().to_int64()
  guard max_bytes > 0L && input <= max_bytes / 2L / output else {
    raise BudgetExceeded
  }
  let weight_bytes = input * output * 2L
  let bias_bytes = output * 2L
  guard bias_bytes <= max_bytes - weight_bytes else { raise BudgetExceeded }
  {
    weight_offset: 0L,
    bias_offset: weight_bytes,
    byte_count: weight_bytes + bias_bytes,
    copies: ReadOnlyArray::from_array([
      {
        tensor_name: "context_embedder.weight",
        source_offset: 0L,
        destination_offset: 0L,
        byte_count: weight_bytes,
      },
      {
        tensor_name: "context_embedder.bias",
        source_offset: 0L,
        destination_offset: weight_bytes,
        byte_count: bias_bytes,
      },
    ]),
  }
}
