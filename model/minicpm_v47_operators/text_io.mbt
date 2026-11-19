///|
/// Untied BF16 embedding/head and centered final norm. Only selected output rows
/// materialize logits. F32 storage carries the BF16-rounded head values exactly.
pub fn DecoderOperators::text_io(
  self : Self,
  maximum_output_rows~ : Int32 = 1,
  maximum_workspace_bytes~ : Int64,
) -> @precision.TextIoPrecision raise {
  let geometry = self.spec.decoder()
  @precision.TextIoPrecision::new(
    vocabulary_size=geometry.vocabulary_size(),
    hidden_width=geometry.hidden_width(),
    stream_count=1,
    maximum_rows=self.maximum_rows,
    maximum_output_rows~,
    normalization_epsilon=self.decoder_norm.epsilon(),
    maximum_workspace_bytes~,
    normalization_law=F32NormalizeThenWeight,
    normalization_weight_law=OnePlusGamma,
    logits_law=Bf16Round,
    head_law=ShapeAwareFloat32,
  )
}
