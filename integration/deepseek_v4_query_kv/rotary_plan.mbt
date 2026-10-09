///|
/// The installed reference disables YaRN on layers without compression.
/// The 64-value simulation block and amax floor are numeric model policy,
/// not a CUDA block size or a persistent FP8-cache representation.
pub fn rotary_kv_plan(
  spec : @spec.DeepSeekV4ModelSpec,
  layer~ : Int,
  maximum_rows~ : Int,
  maximum_workspace_bytes~ : Int64,
) -> @precision.RotaryKvPrecision raise {
  guard layer >= 0 && layer < spec.layer_count() else {
    raise @precision.PrecisionPlanError::IllegalPlacement
  }
  let frequency = match spec.compression_ratio(layer) {
    Some(0) => @precision.RotaryFrequency::Base(theta=spec.rope_theta())
    Some(_) =>
      @precision.RotaryFrequency::Yarn(
        theta=spec.compressed_rope_theta(),
        original_positions=spec.original_position_embeddings(),
        factor=spec.rope_scaling_factor(),
        beta_fast=spec.rope_beta_fast(),
        beta_slow=spec.rope_beta_slow(),
      )
    None => raise @precision.PrecisionPlanError::IllegalPlacement
  }
  @precision.RotaryKvPrecision::new(
    query_heads=spec.attention_head_count(),
    key_value_heads=spec.kv_head_count(),
    head_width=spec.head_dimension(),
    rotary_width=spec.rope_head_dimension(),
    maximum_rows~,
    maximum_positions=spec.max_position_embeddings(),
    frequency~,
    simulation_block_width=64,
    simulation_amax_floor=0.0001,
    simulation_scale=PowerOfTwo,
    maximum_workspace_bytes~,
  )
}
