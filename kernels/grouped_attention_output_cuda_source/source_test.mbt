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
async test "original grouped output replay exports full checkpoint geometry" {
  let plan = @precision.GroupedAttentionOutputPrecision::new(
    head_count=64,
    head_width=512,
    rotary_width=64,
    group_count=8,
    output_rank=1024,
    hidden_width=4096,
    maximum_rows=144,
    maximum_positions=1048576,
    frequency=Base(theta=10000.0),
    maximum_workspace_bytes=33554432L,
  )
  let source = @grouped_attention_output_cuda_source.grouped_attention_output_source(
    plan,
    symbol_prefix="matched_output",
  )
  assert_eq(source.launch(1), (144, 8, 256))
  assert_eq(source.launch(2), (144, 16, 256))
  if @env.get_env_var("LUNAFLUX_MATCHED_OUTPUT_PREVIOUS_SOURCE") is Some(path) {
    // Retained older bundles may have an earlier inverse/output-A generator.
    // Compare the unchanged output-B leaf, not unrelated historical stages.
    let boundary = "extern \"C\" __global__ void matched_output_inverse"
    let current = @utf8.decode(source.source_bytes()).split(boundary).collect()[0].to_owned()
    let previous = @fs.read_file(path).text().split(boundary).collect()[0].to_owned()
    assert_eq(current, previous)
  }
  if @env.get_env_var("LUNAFLUX_MATCHED_OUTPUT_SOURCE_OUTPUT") is Some(path) {
    @fs.write_file(path, source.source_bytes(), create_mode=CreateNew)
  }
  if @env.get_env_var("LUNAFLUX_STRIPED_OUTPUT_SOURCE_OUTPUT") is Some(path) {
    let striped = @precision.GroupedAttentionOutputPrecision::new(
      head_count=64,
      head_width=512,
      rotary_width=64,
      group_count=8,
      output_rank=1024,
      hidden_width=4096,
      maximum_rows=144,
      maximum_positions=1048576,
      frequency=Base(theta=10000.0),
      maximum_workspace_bytes=33554432L,
      accumulation=StripedTreeFloat32(partitions=32),
    )
    let lowered = @grouped_attention_output_cuda_source.grouped_attention_output_source(
      striped,
      symbol_prefix="lowered_output",
    )
    assert_eq(lowered.launch(1), (144, 1024, 256))
    @fs.write_file(path, lowered.source_bytes(), create_mode=CreateNew)
  }
}

///|
test "striped output lowering consumes tree law and reports its actual launch" {
  for partitions in [2, 8, 32, 64] {
    let plan = @precision.GroupedAttentionOutputPrecision::new(
      head_count=2,
      head_width=128,
      rotary_width=64,
      group_count=2,
      output_rank=128,
      hidden_width=128,
      maximum_rows=3,
      maximum_positions=64,
      frequency=Base(theta=10000.0),
      maximum_workspace_bytes=10000L,
      accumulation=StripedTreeFloat32(partitions~),
    )
    if partitions > 32 {
      try
        @grouped_attention_output_cuda_source.grouped_attention_output_source(
          plan,
          symbol_prefix="striped",
        )
      catch {
        @precision.PrecisionPlanError::NoImplementation => ()
        _ => fail("unexpected lowering error")
      } noraise {
        _ => fail("unsupported subgroup law was silently discarded")
      }
    } else {
      let source = @grouped_attention_output_cuda_source.grouped_attention_output_source(
        plan,
        symbol_prefix="striped",
      )
      assert_eq(source.launch(1), (3, 2 * partitions / 2, 256))
      let text = @utf8.decode(source.source_bytes())
      assert_true(
        text.contains("__shfl_down_sync(0xffffffffU, sum, span, \{partitions})"),
      )
      assert_true(text.contains("inner += \{partitions}"))
    }
  }
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
