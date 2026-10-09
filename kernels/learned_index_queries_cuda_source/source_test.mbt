///|
test "index query lowering composes packed and BF16 projections with transform" {
  let plan = @precision.LearnedIndexQueriesPrecision::new(
    hidden_width=128,
    query_rank=128,
    head_count=2,
    @precision.RotaryHadamardFp4Precision::new(
      head_count=2,
      head_width=128,
      rotary_width=64,
      maximum_rows=3,
      maximum_positions=1024,
      frequency=Base(theta=10000.0),
      simulation_block_width=32,
      maximum_workspace_bytes=3072L,
    ),
    maximum_workspace_bytes=4632L,
  )
  let source = @learned_index_queries_cuda_source.learned_index_queries_source(
    plan,
    symbol_prefix="index_fixture",
  )
  let text = @utf8.decode(source.source_bytes())
  assert_true(text.contains("lf_block_activation_scale"))
  assert_true(text.contains("index_fixture_head_projection"))
  assert_true(text.contains("index_fixture_transform_hadamard"))
  assert_true(text.contains("__fmul_rn(__bfloat162float(input[index])"))
  assert_eq(source.launch(0), (3, 1, 256))
  assert_eq(source.launch(1), (1, 1, 256))
  assert_eq(source.launch(2), (1, 1, 256))
}
