///|
test "compact cache lowering consumes group positions without query work" {
  let plan = @precision.RotaryCachePrecision::new(
    head_count=1,
    head_width=512,
    rotary_width=64,
    maximum_rows=2,
    maximum_positions=1048576,
    frequency=Yarn(
      theta=160000.0,
      original_positions=65536,
      factor=16.0,
      beta_fast=32.0,
      beta_slow=1.0,
    ),
    simulation_block_width=64,
    simulation_amax_floor=0.0001,
    simulation_scale=PowerOfTwo,
    maximum_workspace_bytes=2048L,
  )
  let source = @rotary_kv_cuda_source.rotary_cache_source(
    plan,
    symbol_prefix="compact",
  )
  let text = @utf8.decode(source.source_bytes())
  assert_true(text.contains("position = positions[row]"))
  assert_true(text.contains("base + 448ULL + pair * 2ULL"))
  assert_true(text.contains("group >= 7"))
  assert_true(text.contains("powf(160000.0f"))
  assert_true(text.contains("const unsigned bits = __float_as_uint(requested)"))
  assert_false(text.contains("is_query"))
  assert_eq(source.launch(0), (2, 1, 256))
  assert_eq(source.launch(1), (2, 7, 64))
  assert_eq(source.plan().workspace_bytes(), 2048L)
}
