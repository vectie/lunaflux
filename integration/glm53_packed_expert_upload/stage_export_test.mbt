///|
test "export prepared GLM stage source" {
  if @env.get_env_var("LUNA_EXPORT_GLM_STAGE") == Some("1") {
    let first = @string.parse_int(@env.get_env_var("LUNA_GLM_FIRST").unwrap())
    let end = @string.parse_int(@env.get_env_var("LUNA_GLM_END").unwrap())
    let rows = @string.parse_int(@env.get_env_var("LUNA_GLM_ROWS").unwrap())
    let history = @string.parse_int(
      @env.get_env_var("LUNA_GLM_HISTORY").unwrap(),
    )
    let stage = @glm53_packed_expert_upload.GlmDecoderStage::new(
      @glm53_spec.Glm53FlashSpec::official(),
      @pipeline.DecoderStageRange::new(
        layer_count=45,
        first_layer=first,
        end_layer=end,
      ),
      maximum_rows=rows,
      maximum_history=history,
      index_layer_norm_epsilon=1.0e-5,
      maximum_layer_state_bytes=67108864L,
      maximum_layer_workspace_bytes=67108864L,
      maximum_layer_expert_bytes=8589934592L,
      maximum_text_workspace_bytes=627712L,
      accuracy=@precision_ir.Accuracy::new(absolute=0.125, relative=0.125),
    )
    defer stage.close()
    println("LUNA_GLM_STAGE_BEGIN")
    println(@utf8.decode(stage.source_bytes()))
    println("LUNA_GLM_STAGE_END")
  }
}
