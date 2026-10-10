///|
async test "complete hyper envelope preserves controls normalization and both post laws" {
  let text : Array[String] = []
  for
    index, law in [
      @precision.HyperConnectionMixLaw::RoundedBf16ControlsAndProducts,
      OrderedF32SingleBf16Round,
    ] {
    let rendered = @hyper_connection_execution_cuda_source.hyper_connection_source(
      @precision.HyperConnectionPrecision::new(
        stream_count=4,
        hidden_width=6,
        maximum_rows=8,
        sinkhorn_iterations=20,
        control_epsilon=0.000001,
        normalization_epsilon=0.00001,
        mix_law=law,
      ),
      symbol_prefix="hyper" + index.to_string(),
    )
    let source = @utf8.decode(rendered.source_bytes())
    for stage in 0..<4 {
      assert_true(source.contains("void " + rendered.symbol(stage) + "("))
    }
    assert_false(source.contains("lf_hc_stage_sigmoid"))
    assert_true(source.contains("hyper" + index.to_string() + "_sigmoid"))
    assert_eq(rendered.plan().geometry(), (4, 6, 8))
    if index == 0 {
      assert_true(source.contains("mixed_residual = __float2bfloat16_rn"))
    }
    text.push(source)
  }
  // Test-only source export for bounded physical execution, never request JIT.
  if @env.get_env_var("LUNAFLUX_HYPER_SOURCE_OUTPUT") is Some(path) {
    @fs.write_file(path, @utf8.encode(text.join("\n")), create_mode=CreateNew)
  }
}

///|
async test "F32 envelope consumes precision-owned control arithmetic" {
  let rendered = @hyper_connection_execution_cuda_source.hyper_connection_source(
    @precision.HyperConnectionPrecision::new(
      stream_count=4,
      hidden_width=6,
      maximum_rows=3,
      sinkhorn_iterations=20,
      control_epsilon=0.000001,
      normalization_epsilon=0.00001,
      mix_law=TransposedF32ResidualThenBranch,
      control_law=F32ProjectionThenNormalize,
    ),
    symbol_prefix="hyper_f32",
  )
  let source = @utf8.decode(rendered.source_bytes())
  assert_true(source.contains("const float *function_weights"))
  assert_true(source.contains("mixes[output] = __fmul_rn(dot, inverse);"))
  assert_false(source.contains("normalized = __fmul_rn"))
  assert_true(
    source.contains(
      "combination[matrix_base + (unsigned long long)source * 4ULL + (unsigned long long)destination]",
    ),
  )
  assert_true(
    source.contains(
      "const float value = __fadd_rn(__fmul_rn(post_value, branch_value), mixed_residual);",
    ),
  )
  if @env.get_env_var("LUNAFLUX_HYPER_F32_SOURCE_OUTPUT") is Some(path) {
    @fs.write_file(path, rendered.source_bytes(), create_mode=CreateNew)
  }
}
