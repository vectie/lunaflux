///|
test "full workload projection preserves authenticated decoder contract" {
  let decoder = @glm53.build_full_execution_plan(full_metadata())
  let workload = @glm53.build_full_workload_execution_plan(full_metadata())
  assert_eq(workload.kind(), DecoderOnlyText)
  assert_eq(workload.identity(), decoder.identity())
  assert_eq(workload.schema_version(), decoder.schema_version())
  assert_eq(workload.decoder_layer_count(), Some(78))
  assert_eq(workload.maximum_token_context(), Some(1048576))
}

///|
test "Flash workload projection is multimodal and content bound" {
  let decoder = @glm53.build_flash_execution_plan(flash_metadata())
  let workload = @glm53.build_flash_workload_execution_plan(flash_metadata())
  assert_eq(workload.kind(), MultimodalConditionalGeneration)
  assert_eq(workload.identity(), decoder.identity())
  assert_eq(workload.schema_version(), decoder.schema_version())
  assert_eq(workload.decoder_layer_count(), Some(45))
  assert_eq(workload.maximum_token_context(), Some(1048576))
}
