///|
test "Flash sparse decoder composes learned output and both mHC envelopes" {
  let layer = @glm53_packed_expert_upload.GlmSparseLayer::new(
    @glm53_spec.Glm53FlashSpec::official(),
    layer_index=3,
    maximum_queries=2,
    maximum_history=64,
    index_layer_norm_epsilon=1.0e-5,
    maximum_state_bytes=10000000L,
    maximum_workspace_bytes=10000000L,
    maximum_expert_bytes=5000000000L,
    accuracy=@precision_ir.Accuracy::new(absolute=0.125, relative=0.125),
  )
  defer layer.close()
  let source = @utf8.decode(layer.source_bytes())
  for
    symbol in [
      "glm_sparse_3_projection_index_norm", "glm_sparse_3_history_publish", "glm_sparse_3_output",
      "glm_sparse_3_hyper_post", "glm_hyper_ffn_3_norm", "glm_router_3_project",
      "glm_routed_3_gate_up", "glm_moe_combine_3", "glm_hyper_ffn_3_post",
    ] {
    assert_true(source.contains("void " + symbol + "("), msg=symbol)
  }
  assert_true(layer.workspace_bytes() > 0L)
  assert_true(layer.state_bytes() > 0L)
}
