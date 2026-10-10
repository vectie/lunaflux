///|
test "operator observation budget scales with token rows without device effects" {
  for rows in [1, 3, 17] {
    let plan = @text_plan.TextEncoderPlan::new(rows~, epsilon=0.000001)
    let source = @minimax_h3_text_encoder.build_source(plan, prefix="probe")
    // 7 hidden-width snapshots, 4 Q/attention, 4 KV, and one product.
    assert_eq(
      source.first_layer_observation_bytes(),
      rows.to_int64() * 2L * (7L * 5120L + 4L * 8192L + 4L * 1024L + 25600L),
    )
  }
}
