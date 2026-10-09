///|
test "export full checkpoint geometry learned indexer source" {
  if @env.get_env_var("LUNA_EXPORT_LEARNED_INDEXER") != Some("1") {
    return
  }
  let spec = @spec.DeepSeekV4ModelSpec::from_official_model_id(
    "deepseek-ai/DeepSeek-V4-Flash-0731",
  )
  let owner = @deepseek_v4_query_kv.DeepSeekLearnedIndexer::new(
    spec,
    layer=2,
    maximum_rows=3,
    maximum_context=16,
    maximum_state_bytes=1048576L,
    maximum_workspace_bytes=1048576L,
  )
  println("LUNA_LEARNED_INDEXER_SOURCE_BEGIN")
  println(@utf8.decode(owner.source_bytes()))
  println("LUNA_LEARNED_INDEXER_SOURCE_END")
  owner.close()
}
