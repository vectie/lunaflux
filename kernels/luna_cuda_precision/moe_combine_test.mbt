///|
test "MoE output round remains a lowering of generic numeric placement" {
  for
    rounding in [@precision.MoeCombineRounding::SumThenBf16, Bf16RoutedThenSum] {
    let plan = @precision.MoeCombinePrecision::new(
      maximum_rows=3,
      hidden=17,
      rounding~,
      memory_ceiling=510L,
    )
    let source = @luna_cuda_precision.moe_combine_source(
      plan,
      symbol="moe_final",
    )
    assert_eq(source.launch().dimensions(), (1, 256))
    assert_true(source.source().contains("live>3"))
    assert_true(source.source().contains("live*17ULL"))
    assert_eq(
      source.source().contains("__float2bfloat16_rn(routed[i])"),
      rounding == Bf16RoutedThenSum,
    )
    assert_false(source.source().contains("atomicAdd"))
  }
}
