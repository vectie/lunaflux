///|
fn prefix_fixture() -> @precision.QuantizedQueryKvPrecision raise {
  @precision.QuantizedQueryKvPrecision::new(
    hidden_width=256,
    query_rank=128,
    head_count=2,
    head_width=128,
    maximum_rows=3,
    normalization_epsilon=1.0e-6,
    activation_amax_floor=0.0001,
    maximum_workspace_bytes=6144L,
  )
}

///|
test "quantized query prefix composes unique entrypoints and shared FP8 helpers" {
  let source = @quantized_query_kv_cuda_source.quantized_query_kv_source(
    prefix_fixture(),
    symbol_prefix="query_fixture",
  )
  for i in 0..<6 {
    assert_true(@utf8.decode(source.source_bytes()).contains(source.symbol(i)))
  }
  assert_eq(source.launch(2), (3, 1, 256))
  assert_eq(source.launch(3), (3, 2, 256))
  let text = @utf8.decode(source.source_bytes())
  assert_true(text.contains("#ifndef LUNAFLUX_BLOCK_FP8_UE8M0_HELPERS_V1"))
  assert_true(text.contains("__float2bfloat16_rn(__fmul_rn(value, value))"))
  assert_true(text.contains("__float2bfloat16_rn(__fadd_rn(mean,"))
  assert_true(text.contains("__float2bfloat16_rn(rsqrtf(variance))"))
}

///|
test "unimplemented quantization policy is not silently substituted" {
  let plan = @precision.QuantizedQueryKvPrecision::new(
    hidden_width=256,
    query_rank=128,
    head_count=2,
    head_width=128,
    maximum_rows=3,
    normalization_epsilon=1.0e-6,
    activation_amax_floor=0.0,
    maximum_workspace_bytes=6144L,
  )
  try
    @quantized_query_kv_cuda_source.quantized_query_kv_source(
      plan,
      symbol_prefix="query",
    )
  catch {
    @precision.PrecisionPlanError::NoImplementation => ()
    error => raise error
  } noraise {
    _ => fail("different amax policy silently accepted")
  }
}

///|
test "offline quantized query prefix exporter" {
  if @env.get_env_var("LUNA_EXPORT_QUANTIZED_QUERY_KV") == Some("1") {
    println("LUNA_QUERY_KV_SOURCE_BEGIN")
    println(
      @utf8.decode(
        @quantized_query_kv_cuda_source.quantized_query_kv_source(
          prefix_fixture(),
          symbol_prefix="query_fixture",
        ).source_bytes(),
      ),
    )
    println("LUNA_QUERY_KV_SOURCE_END")
  }
}
