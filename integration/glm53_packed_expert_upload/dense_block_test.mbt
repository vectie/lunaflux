///|
test "complete early decoder block includes both envelopes and shared MLP product" {
  let spec = @glm53_spec.Glm53FlashSpec::official()
  let block = @glm53_packed_expert_upload.GlmRecurrentLayer::with_dense_block(
    spec,
    layer_index=2,
    maximum_rows=32,
    maximum_sequences=16,
    cache_slots=32,
    maximum_state_bytes=134217728L,
  )
  defer block.close()
  assert_eq(block.workspace_bytes(), 148669440L)
  let source = @utf8.decode(block.source_bytes())
  for
    symbol in [
      "glm_hyper_attn_2_post", "glm_hyper_ffn_2_norm", "glm_dense_ffn_2_gate_up",
      "glm_dense_ffn_2_down", "glm_hyper_ffn_2_post",
    ] {
    assert_true(source.contains("void " + symbol + "("))
  }
  try
    @glm53_packed_expert_upload.GlmRecurrentLayer::with_dense_block(
      spec,
      layer_index=4,
      maximum_rows=32,
      maximum_sequences=16,
      cache_slots=32,
      maximum_state_bytes=134217728L,
    )
  catch {
    @precision_ir.PrecisionPlanError::IllegalPlacement => ()
    e => raise e
  } noraise {
    _ => fail("MoE layer silently replaced with dense FFN")
  }
}
