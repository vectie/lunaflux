///|
fn fixture() -> @precision.GroupedAttentionOutputPrecision raise {
  @precision.GroupedAttentionOutputPrecision::new(
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
  )
}

///|
test "grouped output has inverse rotation and separate BF16 versus quantized projection" {
  let source = @grouped_attention_output_cuda_source.grouped_attention_output_source(
    fixture(),
    symbol_prefix="output_test",
  )
  assert_eq(source.launch(0), (3, 2, 256))
  assert_eq(source.launch(1), (3, 2, 256))
  assert_eq(source.launch(2), (3, 1, 256))
  let code = @utf8.decode(source.source_bytes())
  assert_true(code.contains("sine = -sinf(angle)"))
  assert_true(
    code.contains("__float2bfloat16_rn(__fmul_rn(lf_block_e4m3_decode"),
  )
  assert_true(code.contains("lf_block_activation_scale"))
}

///|
test "offline grouped output exporter" {
  if @env.get_env_var("LUNA_EXPORT_GROUPED_OUTPUT") == Some("1") {
    let source = @grouped_attention_output_cuda_source.grouped_attention_output_source(
      fixture(),
      symbol_prefix="output",
    )
    println("LUNA_GROUPED_OUTPUT_SOURCE_BEGIN")
    println(@utf8.decode(source.source_bytes()))
    let yarn = @precision.GroupedAttentionOutputPrecision::new(
      head_count=2,
      head_width=128,
      rotary_width=64,
      group_count=2,
      output_rank=128,
      hidden_width=128,
      maximum_rows=3,
      maximum_positions=1048576,
      frequency=Yarn(
        theta=160000.0,
        original_positions=65536,
        factor=16.0,
        beta_fast=32.0,
        beta_slow=1.0,
      ),
      maximum_workspace_bytes=10000L,
    )
    println(
      @utf8.decode(
        @grouped_attention_output_cuda_source.grouped_attention_output_source(
          yarn,
          symbol_prefix="output_yarn",
        ).source_bytes(),
      ),
    )
    println("LUNA_GROUPED_OUTPUT_SOURCE_END")
  }
}
