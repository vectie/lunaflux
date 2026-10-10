///|
pub struct TextEncoderSource {
  plan : @text_plan.TextEncoderPlan
  symbols : ReadOnlyArray[String]
  source : Bytes
}

///|
pub fn build_source(
  plan : @text_plan.TextEncoderPlan,
  prefix~ : String,
) -> TextEncoderSource raise {
  let symbols : ReadOnlyArray[String] = [
    prefix + "_embedding",
    prefix + "_norm",
    prefix + "_q",
    prefix + "_kv",
    prefix + "_qknorm",
    prefix + "_rotary",
    prefix + "_attention",
    prefix + "_out",
    prefix + "_mlp_gate_up",
    prefix + "_mlp_down",
    prefix + "_residual",
    prefix + "_visual_set",
    prefix + "_visual_add",
  ]
  let embedding = @decoder_foundation_cuda_source.token_embedding_bf16_source(
    vocabulary_size=151936,
    hidden_size=5120,
    function_symbol=symbols[0],
  ).source_bytes()
  let norm = @text_encoder_cuda_source.norm_source(
    rows=plan.rows,
    width=5120,
    epsilon=plan.epsilon,
    symbol=symbols[1],
  )
  let q = @decoder_projection_cuda_source.dense_projection_bf16_source(
    input_width=5120,
    output_width=8192,
    maximum_rows=plan.rows,
    function_symbol=symbols[2],
  ).source_bytes()
  let kv = @decoder_projection_cuda_source.dense_projection_bf16_source(
    input_width=5120,
    output_width=1024,
    maximum_rows=plan.rows,
    function_symbol=symbols[3],
  ).source_bytes()
  let att = @text_encoder_cuda_source.attention_source(
    rows=plan.rows,
    query_heads=64,
    kv_heads=8,
    dimension=128,
    epsilon=plan.epsilon,
    prefix~,
  )
  let out = @decoder_projection_cuda_source.dense_projection_bf16_source(
    input_width=8192,
    output_width=5120,
    maximum_rows=plan.rows,
    function_symbol=symbols[7],
  ).source_bytes()
  let mlp = @gated_mlp_cuda_source.staged_packed_swiglu_bf16_source(
    rows=plan.rows,
    hidden_size=5120,
    intermediate_size=25600,
    function_prefix=prefix + "_mlp",
  )
  let residual = @decoder_foundation_cuda_source.residual_add_bf16_source(
    elements=plan.rows.to_int64() * 5120L,
    function_symbol=symbols[10],
  )
  {
    plan,
    symbols,
    source: @cuda_translation_unit.compose([
      embedding,
      norm,
      q,
      kv,
      att,
      out,
      mlp,
      residual,
      @text_encoder_cuda_source.visual_scatter_source(
        rows=plan.rows,
        width=5120,
        prefix~,
      ),
    ]),
  }
}
