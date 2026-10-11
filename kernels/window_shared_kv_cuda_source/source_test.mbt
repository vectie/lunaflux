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
  assert_true(!@utf8.decode(source.source_bytes()).contains("}#include"))
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

///|
async test "original window replay exports literal full-head rotary and online probability boundaries" {
  let rotary = @precision.RotaryKvPrecision::new(
    query_heads=64,
    key_value_heads=1,
    head_width=512,
    rotary_width=64,
    maximum_rows=144,
    maximum_positions=1048576,
    frequency=Base(theta=10000.0),
    simulation_block_width=64,
    simulation_amax_floor=0.0001,
    simulation_scale=PowerOfTwo,
    maximum_workspace_bytes=33554432L,
  )
  let window = @precision.WindowSharedKvPrecision::new(
    query_heads=64,
    head_width=512,
    window_tokens=128,
    maximum_rows=144,
    maximum_positions=1048576,
    maximum_state_bytes=33554432L,
    probability=OnlineBf16(block_tokens=64),
  )
  let inverse = @precision.RotaryTensorPrecision::new(
    head_count=64,
    head_width=512,
    rotary_width=64,
    maximum_rows=144,
    maximum_positions=1048576,
    frequency=Base(theta=10000.0),
    inverse=true,
    maximum_workspace_bytes=33554432L,
  )
  let source = @rotary_source.rotary_kv_source(
      rotary,
      symbol_prefix="matched_window",
    ).source_bytes() +
    b"\n" +
    @window_shared_kv_cuda_source.window_shared_kv_source(
      window,
      symbol_prefix="matched_window",
    ).source_bytes() +
    b"\n" +
    @utf8.encode(
      @rotary_source.rotary_tensor_source(
        inverse,
        symbol="matched_window_inverse",
      ),
    )
  assert_eq(window.probability_law(), OnlineBf16(block_tokens=64))
  assert_true(@utf8.decode(source).contains("window_span = base == 0"))
  if @env.get_env_var("LUNAFLUX_MATCHED_WINDOW_SOURCE_OUTPUT") is Some(path) {
    @fs.write_file(path, source, create_mode=CreateNew)
  }
}
