///|
test "output B law owns its actual launch independently of output A" {
  let plan = @precision.GroupedAttentionOutputPrecision::new(
    head_count=2,
    head_width=128,
    rotary_width=64,
    group_count=2,
    output_rank=128,
    hidden_width=128,
    maximum_rows=3,
    maximum_positions=1048576,
    frequency=Base(theta=10000.0),
    maximum_workspace_bytes=10000L,
    projection_scaled_dot_law=UnscaledBlockMatrixFloat32(
      block_columns=128,
      publication=SeparateMultiplyAdd,
    ),
  )
  let source = @grouped_attention_output_cuda_source.grouped_attention_output_source(
    plan,
    symbol_prefix="matrix_output",
  )
  assert_eq(source.launch(0), (3, 2, 256))
  assert_eq(source.launch(1), (3, 2, 256))
  assert_eq(source.launch(2), (3, 4, 128))
  let text = @utf8.decode(source.source_bytes())
  assert_true(text.contains("matrix_output_output_block_dot<128,256>"))
  assert_true(text.contains("float lf_block_e4m3_decode"))
  assert_true(text.contains("float lf_block_e8m0_decode"))
}
