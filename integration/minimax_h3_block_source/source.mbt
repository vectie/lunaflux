///|
pub struct BlockSource {
  source : Bytes
  symbols : ReadOnlyArray[String]
  rows : Int
  distinct_timesteps : Int
  shape : @joint_diffusion_plan.JointDiffusionDenoiserShape
  model : @joint_diffusion_plan.JointDiffusionPlan
  packed_mapping : Bytes
}

///|
pub(all) suberror BlockSourceError {
  InvalidShape
}

///|
/// Offline source assembly only; no request-path compiler or device loading.
/// Symbols are parameter, RMSNorm, affine, residual, QKV, QK norm, attention,
/// output projection, gate/up and down. Shared activation is emitted last.
pub fn build(
  model : @joint_diffusion_plan.JointDiffusionPlan,
  sequence : @minimax_h3_sequence.Sequence,
  distinct_timesteps~ : Int,
  block_norm_epsilon~ : Double,
  function_prefix~ : String,
) -> BlockSource raise {
  let shape = model.denoiser()
  let rows = sequence.assembly().rows().length()
  guard sequence.assembly().width() == shape.hidden_size() &&
    distinct_timesteps > 0 &&
    distinct_timesteps <= 715827882 &&
    !block_norm_epsilon.is_nan() &&
    !block_norm_epsilon.is_inf() &&
    block_norm_epsilon > 0.0 else {
    raise InvalidShape
  }
  let hidden = shape.hidden_size()
  let parameter_rows = distinct_timesteps * 3
  let symbols = [
    function_prefix + "_parameters",
    function_prefix + "_norm",
    function_prefix + "_mod_affine",
    function_prefix + "_mod_residual",
    function_prefix + "_attn_qkv",
    function_prefix + "_attn_qk_norm",
    function_prefix + "_attn_attention",
    function_prefix + "_attn_output",
    function_prefix + "_mlp_gate_up",
    function_prefix + "_mlp_down",
    function_prefix + "_conditioning",
  ]
  let attention = @joint_attention_cuda_source.staged_joint_attention_bf16_source(
    rows~,
    hidden_size=hidden,
    head_count=shape.attention_head_count(),
    head_dimension=shape.attention_head_dimension(),
    rotary_dimension=shape.attention_rotary_dimension(),
    epsilon_numerator=shape.query_key_norm_epsilon_numerator(),
    epsilon_denominator=shape.query_key_norm_epsilon_denominator(),
    apply_rotary=true,
    function_prefix=function_prefix + "_attn",
    document_boundaries=sequence.attention_boundaries(),
  )
  let norm = @decoder_foundation_cuda_source.rms_norm_bf16_fixed_rows_source(
    rows=rows.to_int64(),
    hidden_size=hidden,
    epsilon=block_norm_epsilon,
    function_symbol=symbols[1],
  )
  let mlp = @gated_mlp_cuda_source.staged_packed_swiglu_bf16_source(
    rows~,
    hidden_size=hidden,
    intermediate_size=shape.feed_forward_dimension(),
    function_prefix=function_prefix + "_mlp",
  )
  let modulation = @affine_modulation_cuda_source.indexed_bf16_modulation_source(
    rows~,
    width=hidden,
    parameter_rows~,
    function_prefix=function_prefix + "_mod",
  )
  let parameters = @affine_modulation_cuda_source.grouped_parameter_projection_source(
    rows=distinct_timesteps,
    input_width=shape.final_adaln_conditioning_dimension(),
    width=hidden,
    groups=3,
    fields=6,
    function_symbol=symbols[0],
  )
  let activation = @affine_modulation_cuda_source.f32_silu_to_bf16_source(
    elements=distinct_timesteps.to_int64() *
      shape.final_adaln_conditioning_dimension().to_int64(),
    function_symbol=symbols[10],
  )
  {
    source: @cuda_translation_unit.compose([
      attention.source_bytes(),
      norm.source_bytes(),
      mlp,
      modulation,
      parameters,
      activation,
    ]),
    symbols: ReadOnlyArray::from_array(symbols),
    rows,
    distinct_timesteps,
    shape,
    model,
    packed_mapping: sequence.assembly().mapping(),
  }
}
