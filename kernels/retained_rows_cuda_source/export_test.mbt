///|
test "offline retained rows numerical fixture exporter" {
  if @env.get_env_var("LUNA_EXPORT_RETAINED_ROWS") != Some("1") {
    return
  }
  println("LUNA_RETAINED_ROWS_SOURCE_BEGIN")
  for capacity in [3, 0] {
    let plan = @precision.RetainedRowsPrecision::new(
      word_width=16,
      position_stride=4,
      maximum_input_rows=2,
      maximum_stored_rows=capacity,
      maximum_state_bytes=1048576L,
      maximum_workspace_bytes=12L,
    )
    println(
      @utf8.decode(
        @retained_rows_cuda_source.retained_rows_source(
          plan,
          symbol_prefix="retained\{capacity}",
        ).source_bytes(),
      ),
    )
  }
  println("LUNA_RETAINED_ROWS_SOURCE_END")
}
