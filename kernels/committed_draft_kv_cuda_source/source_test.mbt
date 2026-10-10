///|
test "prime and prediction preserve distinct publication and noncausal read sets" {
  let plan = @precision.CommittedDraftKvPrecision::new(
    query_heads=2,
    head_width=8,
    window_tokens=4,
    maximum_main_rows=8,
    maximum_draft_rows=3,
    maximum_positions=32,
    maximum_state_bytes=184L,
  )
  let prime = @committed_draft_kv_cuda_source.committed_draft_kv_source(
    plan,
    phase=Prime,
    symbol_prefix="prime",
  )
  let predict = @committed_draft_kv_cuda_source.committed_draft_kv_source(
    plan,
    phase=Predict,
    symbol_prefix="predict",
  )
  assert_eq(prime.stage_count(), 3)
  assert_eq(predict.stage_count(), 4)
  assert_eq(predict.symbol(1), "predict_copy")
  assert_eq(predict.symbol(2), "predict_publish")
  assert_eq(predict.symbol(3), "predict_attention")
  assert_eq(predict.launch(3), (3, 2, 128))
  let text = @utf8.decode(predict.source_bytes())
  assert_true(text.contains("length = main + descriptor[2]"))
  assert_true(text.contains("main != 1"))
  assert_true(!text.contains("draft_kv[(unsigned long long)(latest"))
  if @env.get_env_var("LUNA_EXPORT_COMMITTED_DRAFT") == Some("1") {
    println("LUNA_MAIN_BEGIN")
    println(@utf8.decode(prime.source_bytes()))
    println(text)
    println("LUNA_MAIN_END")
  }
}
