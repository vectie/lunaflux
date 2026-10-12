///|
test "explicit router arithmetic reaches both weight representations and selections" {
  for format in [@format.Format::Bf16, F32] {
    for
      selection in [
        @precision.MoeRoutingSelection::GroupedTopK,
        TokenHash(vocabulary_size=3),
      ] {
      let source = @moe_routing_cuda_source.routing_source(
        routing_fixture(
          format,
          SqrtSoftplus,
          selection~,
          projection_law=GroupedVectorFloat32(
            groups=1,
            partitions=16,
            vector_width=1,
            publication_partitions=1,
            multiply_add=FusedMultiplyAdd,
          ),
          softplus_law=ThresholdedFloat32(threshold=18.0),
          selected_sum_law=ButterflyFloat32,
        ),
        symbol="arithmetic_router",
      )
      assert_eq(source.launches()[0], (8, 16))
      assert_eq(
        source.plan().workspace_bytes(),
        if selection is GroupedTopK {
          128L
        } else {
          96L
        },
      )
      let text = @utf8.decode(source.source_bytes())
      assert_true(text.contains("__fmaf_rn("))
      assert_true(text.contains("x>18.0f ? x : log1pf(expf(x))"))
      assert_true(text.contains("selected_partial[2]"))
      assert_true(text.contains("span=8"))
    }
  }
}

///|
test "unsupported dot alternative rejects instead of generating the ordered kernel" {
  try
    @moe_routing_cuda_source.routing_source(
      routing_fixture(Bf16, SqrtSoftplus, projection_law=CompensatedFloat32),
      symbol="not_supported",
    )
  catch {
    @precision.PrecisionPlanError::IllegalPlacement => ()
    _ => fail("unexpected unsupported law error")
  } noraise {
    _ => fail("silent dot fallback")
  }
}

///|
test "six selected slots use neutral tree padding without extra logical reads" {
  let text = @grouped.selected_weight_butterfly_source(
    width=6,
    value_expression="values[slot]",
    output_symbol="sum",
  )
  assert_true(text.contains("selected_partial[8]"))
  assert_true(text.contains("slot<6 ? values[slot] : 0.0f"))
  assert_true(text.contains("span=4"))
  try
    @grouped.selected_weight_butterfly_source(
      width=0,
      value_expression="values[slot]",
      output_symbol="sum",
    )
  catch {
    _ => ()
  } noraise {
    _ => fail("empty padded reduction")
  }
}
