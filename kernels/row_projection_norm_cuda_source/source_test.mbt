///|
test "row projection normalization lowers two ordered shared kernels" {
  let plan = @precision.RowProjectionNormPrecision::new(
    input_width=256,
    output_width=128,
    maximum_rows=3,
    normalization_epsilon=1.0e-6,
    activation_amax_floor=0.0001,
    maximum_workspace_bytes=1536L,
  )
  let source = @row_projection_norm_cuda_source.row_projection_norm_source(
    plan,
    symbol_prefix="main_fixture",
  )
  assert_eq(source.launch(0), (3, 1, 256))
  assert_eq(source.launch(1), (3, 1, 256))
  assert_eq(source.symbol(0), "main_fixture_projection")
  assert_eq(source.symbol(1), "main_fixture_norm")
  let text = @utf8.decode(source.source_bytes())
  assert_true(text.contains("lf_block_activation_scale"))
  assert_true(text.contains("main_fixture_norm"))
  try
    @row_projection_norm_cuda_source.row_projection_norm_source(
      plan,
      symbol_prefix="bad-symbol",
    )
  catch {
    _ => ()
  } noraise {
    _ => fail("unsafe symbol accepted")
  }
}

///|
test "export actual row projection normalization source" {
  if @env.get_env_var("LUNA_EXPORT_ROW_PROJECTION_NORM") != Some("1") {
    return
  }
  let plan = @precision.RowProjectionNormPrecision::new(
    input_width=256,
    output_width=128,
    maximum_rows=3,
    normalization_epsilon=1.0e-6,
    activation_amax_floor=0.0001,
    maximum_workspace_bytes=1536L,
  )
  println("LUNA_MAIN_BEGIN")
  println(
    @utf8.decode(
      @row_projection_norm_cuda_source.row_projection_norm_source(
        plan,
        symbol_prefix="main_fixture",
      ).source_bytes(),
    ),
  )
  println("LUNA_MAIN_END")
}
