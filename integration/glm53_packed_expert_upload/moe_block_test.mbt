///|
test "recurrent MoE block composes both envelopes and packed/shared expert stages" {
  let spec = @glm53_spec.Glm53FlashSpec::official()
  let block = @glm53_packed_expert_upload.GlmRecurrentLayer::with_moe_block(
    spec,
    layer_index=4,
    maximum_rows=2,
    maximum_sequences=2,
    cache_slots=2,
    maximum_state_bytes=16777216L,
    maximum_expert_bytes=5000000000L,
    accuracy=@precision_ir.Accuracy::new(absolute=0.125, relative=0.125),
  )
  defer block.close()
  let source = @utf8.decode(block.source_bytes())
  for
    symbol in [
      "glm_recurrent_4_15", "glm_hyper_attn_4_post", "glm_hyper_ffn_4_norm", "glm_routed_4_gate_up",
      "glm_shared_4_gate_up", "glm_moe_combine_4", "glm_hyper_ffn_4_post",
    ] {
    assert_true(source.contains("void " + symbol + "("))
  }
  assert_true(block.workspace_bytes() > 0L)
  for layer in [0, 3] {
    try
      @glm53_packed_expert_upload.GlmRecurrentLayer::with_moe_block(
        spec,
        layer_index=layer,
        maximum_rows=2,
        maximum_sequences=2,
        cache_slots=2,
        maximum_state_bytes=16777216L,
        maximum_expert_bytes=5000000000L,
        accuracy=@precision_ir.Accuracy::new(absolute=0.125, relative=0.125),
      )
    catch {
      @precision_ir.PrecisionPlanError::IllegalPlacement => ()
      e => raise e
    } noraise {
      _ => fail("dense/DSA attention silently replaced")
    }
  }
}
