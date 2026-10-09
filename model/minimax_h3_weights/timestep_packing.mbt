///|
pub struct TimestepWeightPacking {
  first_weight_offset : Int64
  first_bias_offset : Int64
  second_weight_offset : Int64
  second_bias_offset : Int64
  byte_count : Int64
  copies : ReadOnlyArray[BlockWeightCopy]
}

///|
pub fn timestep_weight_packing(
  model : @joint_diffusion_plan.JointDiffusionPlan,
  max_bytes~ : Int64,
) -> TimestepWeightPacking raise BlockPackingError {
  let shape = model.denoiser()
  let input = shape.timestep_frequency_dimension().to_int64()
  let hidden = shape.timestep_embedding_hidden_dimension().to_int64()
  let output = shape.final_adaln_conditioning_dimension().to_int64()
  guard max_bytes > 0L &&
    hidden <= max_bytes / 4L / input &&
    hidden <= max_bytes / 4L / output else {
    raise BudgetExceeded
  }
  let sizes = [
    input * hidden * 4L,
    hidden * 4L,
    hidden * output * 4L,
    output * 4L,
  ]
  let names = [
    "time_embedder.linear_1.weight", "time_embedder.linear_1.bias", "time_embedder.linear_2.weight",
    "time_embedder.linear_2.bias",
  ]
  let copies : Array[BlockWeightCopy] = []
  let mut offset = 0L
  for i, size in sizes {
    guard size <= max_bytes - offset else { raise BudgetExceeded }
    copies.push({
      tensor_name: names[i],
      source_offset: 0L,
      destination_offset: offset,
      byte_count: size,
    })
    offset += size
  }
  {
    first_weight_offset: 0L,
    first_bias_offset: copies[1].destination_offset,
    second_weight_offset: copies[2].destination_offset,
    second_bias_offset: copies[3].destination_offset,
    byte_count: offset,
    copies: ReadOnlyArray::from_array(copies),
  }
}
