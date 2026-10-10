///|
test "all profiles select F32 projected controls and transposed residual semantics" {
  for
    name in [
      "deepseek-ai/DeepSeek-V4-Flash", "deepseek-ai/DeepSeek-V4-Pro", "deepseek-ai/DeepSeek-V4-Flash-0731",
    ] {
    let spec = @spec.DeepSeekV4ModelSpec::from_official_model_id(name)
    let plan = @deepseek_v4_hyper_connection.hyper_connection_plan(
      spec,
      maximum_rows=3,
    )
    assert_true(plan.control_law() is F32ProjectionThenNormalize)
    assert_true(plan.mix_law() is TransposedF32ResidualThenBranch)
    assert_eq(
      plan.function_weight_bytes(),
      24L * 4L * spec.hidden_size().to_int64() * 4L,
    )
    for
      branch in [
        @deepseek_v4_hyper_connection.DeepSeekHyperBranch::Attention,
        FeedForward,
      ] {
      let owner = @deepseek_v4_hyper_connection.DeepSeekHyperConnection::new(
        spec,
        layer=3,
        branch~,
        maximum_rows=3,
      )
      let source = @utf8.decode(owner.source_bytes())
      assert_true(source.contains("const float *function_weights"))
      assert_true(source.contains("mixes[output] = __fmul_rn(dot, inverse);"))
      assert_true(
        source.contains(
          "(unsigned long long)source * 4ULL + (unsigned long long)destination",
        ),
      )
      owner.close()
    }
  }
}
