///|
pub struct RefinerSource {
  source : Bytes
  symbols : ReadOnlyArray[String]
  model : @joint_diffusion_plan.JointDiffusionPlan
  rows : Int
}

///|
pub fn build_refiner(
  model : @joint_diffusion_plan.JointDiffusionPlan,
  text_rows~ : Int,
  block_norm_epsilon~ : Double,
  function_prefix~ : String,
) -> RefinerSource raise {
  let s = model.denoiser()
  let symbols = [
    function_prefix + "_norm",
    function_prefix + "_final_norm",
    function_prefix + "_residual",
    function_prefix + "_attn_qkv",
    function_prefix + "_attn_qk_norm",
    function_prefix + "_attn_attention",
    function_prefix + "_attn_output",
    function_prefix + "_mlp_gate_up",
    function_prefix + "_mlp_down",
    function_prefix + "_context",
  ]
  let norm = @decoder_foundation_cuda_source.rms_norm_bf16_fixed_rows_source(
    rows=text_rows.to_int64(),
    hidden_size=s.hidden_size(),
    epsilon=block_norm_epsilon,
    function_symbol=symbols[0],
  ).source_bytes()
  let final_norm = @decoder_foundation_cuda_source.rms_norm_bf16_fixed_rows_source(
    rows=text_rows.to_int64(),
    hidden_size=s.hidden_size(),
    epsilon=s.final_norm_epsilon(),
    function_symbol=symbols[1],
  ).source_bytes()
  let residual = @decoder_foundation_cuda_source.residual_add_bf16_source(
    elements=text_rows.to_int64() * s.hidden_size().to_int64(),
    function_symbol=symbols[2],
  )
  let attention = @joint_attention_cuda_source.staged_joint_attention_bf16_source(
    rows=text_rows,
    hidden_size=s.hidden_size(),
    head_count=s.attention_head_count(),
    head_dimension=s.attention_head_dimension(),
    rotary_dimension=0,
    epsilon_numerator=s.query_key_norm_epsilon_numerator(),
    epsilon_denominator=s.query_key_norm_epsilon_denominator(),
    apply_rotary=false,
    function_prefix=function_prefix + "_attn",
  ).source_bytes()
  let mlp = @gated_mlp_cuda_source.staged_packed_swiglu_bf16_source(
    rows=text_rows,
    hidden_size=s.hidden_size(),
    intermediate_size=s.feed_forward_dimension(),
    function_prefix=function_prefix + "_mlp",
  )
  let context = @affine_modulation_cuda_source.grouped_parameter_projection_source(
    rows=text_rows,
    input_width=s.text_conditioning_dimension(),
    width=s.hidden_size(),
    groups=1,
    fields=1,
    function_symbol=symbols[9],
  )
  {
    source: @cuda_translation_unit.compose([
      norm, final_norm, residual, attention, mlp, context,
    ]),
    symbols: ReadOnlyArray::from_array(symbols),
    model,
    rows: text_rows,
  }
}
