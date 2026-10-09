///|
async test "dense executable stages share product and bound live rows" {
  let rendered = @dense_mlp_execution_cuda_source.dense_mlp_source(
    @precision.DenseSwiGluPrecision::new(
      hidden_width=6,
      intermediate_width=9,
      maximum_rows=8,
    ),
    symbol_prefix="dense",
  )
  let source = @utf8.decode(rendered.source_bytes())
  for stage in 0..<2 {
    assert_true(source.contains("void " + rendered.symbol(stage) + "("))
  }
  assert_true(source.contains("rows=counts[3]"))
  assert_true(source.contains("product[row*9ULL+inner]"))
  assert_eq(rendered.plan().workspace_bytes(), 144L)
  if @env.get_env_var("LUNAFLUX_DENSE_SOURCE_OUTPUT") is Some(path) {
    @fs.write_file(path, rendered.source_bytes(), create_mode=CreateNew)
  }
}
