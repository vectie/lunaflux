///|
fn fixture(
  frequency : @precision.RotaryFrequency,
  scale : @precision.SimulatedCacheScale,
) -> @precision.RotaryKvPrecision raise {
  @precision.RotaryKvPrecision::new(
    query_heads=2,
    key_value_heads=1,
    head_width=128,
    rotary_width=64,
    maximum_rows=3,
    maximum_positions=1048576,
    frequency~,
    simulation_block_width=64,
    simulation_amax_floor=0.0001,
    simulation_scale=scale,
    maximum_workspace_bytes=2304L,
  )
}

///|
test "full-head rotary lowering preserves prefix and limits KV simulation to non-rotary blocks" {
  let source = @rotary_kv_cuda_source.rotary_kv_source(
    fixture(Base(theta=10000.0), PowerOfTwo),
    symbol_prefix="rotary_fixture",
  )
  let text = @utf8.decode(source.source_bytes())
  assert_true(
    text.contains("output_words[base + col] = input_words[base + col]"),
  )
  assert_true(text.contains("base + 64ULL + pair * 2ULL"))
  assert_true(text.contains("group >= 1"))
  assert_true(text.contains("powf(10000.0f"))
  assert_false(text.contains("correction"))
  assert_eq(source.launch(0), (3, 3, 256))
  assert_eq(source.launch(1), (3, 1, 64))
  assert_true(text.contains("__nv_cvt_float_to_fp8"))
  assert_true(text.contains("const unsigned bits = __float_as_uint(requested)"))
  let yarn = @rotary_kv_cuda_source.rotary_kv_source(
    fixture(
      Yarn(
        theta=160000.0,
        original_positions=65536,
        factor=16.0,
        beta_fast=32.0,
        beta_slow=1.0,
      ),
      LinearF32,
    ),
    symbol_prefix="yarn_fixture",
  )
  assert_true(@utf8.decode(yarn.source_bytes()).contains("powf(160000.0f"))
  assert_true(
    @utf8.decode(yarn.source_bytes()).contains("const float scale = requested;"),
  )
  assert_false(
    @utf8.decode(yarn.source_bytes()).contains("__float2bfloat16_rn(cosf"),
  )
}

///|
test "offline rotary simulation exporter" {
  if @env.get_env_var("LUNA_EXPORT_ROTARY_KV") == Some("1") {
    println("LUNA_ROTARY_KV_SOURCE_BEGIN")
    for
      (frequency, scale, prefix) in [
        (
          @precision.RotaryFrequency::Base(theta=10000.0),
          @precision.SimulatedCacheScale::PowerOfTwo,
          "rotary_fixture",
        ),
        (
          @precision.RotaryFrequency::Yarn(
            theta=160000.0,
            original_positions=65536,
            factor=16.0,
            beta_fast=32.0,
            beta_slow=1.0,
          ),
          @precision.SimulatedCacheScale::PowerOfTwo,
          "yarn_fixture",
        ),
        (
          @precision.RotaryFrequency::Base(theta=10000.0),
          @precision.SimulatedCacheScale::LinearF32,
          "linear_fixture",
        ),
      ] {
      println(
        @utf8.decode(
          @rotary_kv_cuda_source.rotary_kv_source(
            @precision.RotaryKvPrecision::new(
              query_heads=2,
              key_value_heads=1,
              head_width=512,
              rotary_width=64,
              maximum_rows=3,
              maximum_positions=1048576,
              frequency~,
              simulation_block_width=64,
              simulation_amax_floor=0.0001,
              simulation_scale=scale,
              maximum_workspace_bytes=9216L,
            ),
            symbol_prefix=prefix,
          ).source_bytes(),
        ),
      )
    }
    println("LUNA_ROTARY_KV_SOURCE_END")
  }
}
