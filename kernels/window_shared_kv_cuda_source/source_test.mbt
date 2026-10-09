///|
test "window attention source delays cache overwrite until every row retires" {
  let plan = @precision.WindowSharedKvPrecision::new(
    query_heads=2,
    head_width=512,
    window_tokens=128,
    maximum_rows=256,
    maximum_positions=1048576,
    maximum_state_bytes=1048576L,
  )
  let source = @window_shared_kv_cuda_source.window_shared_kv_source(
    plan,
    symbol_prefix="window",
  )
  assert_eq(source.launch(1), (256, 2, 128))
  assert_eq(source.launch(2), (128, 1, 128))
  assert_eq(source.symbol(3), "window_publish")
  assert_true(source.source_bytes().length() > 2000)
}

///|
test "offline shared window exporter" {
  if @env.get_env_var("LUNA_EXPORT_SHARED_WINDOW") != Some("1") {
    return
  }
  let plan = @precision.WindowSharedKvPrecision::new(
    query_heads=2,
    head_width=512,
    window_tokens=128,
    maximum_rows=256,
    maximum_positions=1048576,
    maximum_state_bytes=1048576L,
  )
  let source = @window_shared_kv_cuda_source.window_shared_kv_source(
    plan,
    symbol_prefix="window",
  )
  println("LUNA_SHARED_WINDOW_SOURCE_BEGIN")
  println(@utf8.decode(source.source_bytes()))
  println("LUNA_SHARED_WINDOW_SOURCE_END")
}
