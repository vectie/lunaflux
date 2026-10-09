///|
pub struct TimestepSource {
  source : Bytes
  symbols : ReadOnlyArray[String]
  rows : Int
  input_width : Int
  hidden_width : Int
  output_width : Int
}

///|
pub fn build_timestep(
  model : @joint_diffusion_plan.JointDiffusionPlan,
  distinct_timesteps~ : Int,
  function_prefix~ : String,
) -> TimestepSource raise {
  let shape = model.denoiser()
  let rows = distinct_timesteps
  let input_width = shape.timestep_frequency_dimension()
  let hidden_width = shape.timestep_embedding_hidden_dimension()
  let output_width = shape.final_adaln_conditioning_dimension()
  let symbols = [
    function_prefix + "_frequency",
    function_prefix + "_mlp_hidden",
    function_prefix + "_mlp_output",
    function_prefix + "_activation",
  ]
  let frequency = @timestep_embedding_cuda_source.cosine_sine_timestep_rows_source(
    rows~,
    frequency_width=input_width,
    function_symbol=symbols[0],
  ).source_bytes()
  let mlp = @timestep_embedding_cuda_source.staged_dense_silu_dense_f32_rows_source(
    rows~,
    input_width~,
    hidden_width~,
    output_width~,
    function_prefix=function_prefix + "_mlp",
  )
  let activation = @affine_modulation_cuda_source.f32_silu_to_bf16_source(
    elements=rows.to_int64() * output_width.to_int64(),
    function_symbol=symbols[3],
  )
  {
    source: frequency + mlp + activation,
    symbols: ReadOnlyArray::from_array(symbols),
    rows,
    input_width,
    hidden_width,
    output_width,
  }
}
