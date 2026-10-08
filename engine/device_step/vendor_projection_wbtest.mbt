///|
test "vendor projection exact domain excludes decode, empty, tail and padded bucket work" {
  let summary = empty_device_step_summary()
  assert_false(vendor_projection_matches(2048, summary))
  for live in [1, 16, 1024, 2047, 2049] {
    assert_false(
      vendor_projection_matches(2048, {
        ..summary,
        prefill_count: 1,
        token_count: live,
      }),
    )
  }
  assert_false(
    vendor_projection_matches(2048, {
      ..summary,
      decode_count: 16,
      token_count: 2048,
    }),
  )
  assert_true(
    vendor_projection_matches(2048, {
      ..summary,
      prefill_count: 1,
      token_count: 2048,
    }),
  )
  assert_true(
    vendor_projection_matches(2048, {
      ..summary,
      prefill_count: 1,
      decode_count: 8,
      token_count: 2048,
    }),
  )
  assert_false(
    vendor_projection_matches(0, {
      ..summary,
      prefill_count: 1,
      token_count: 2048,
    }),
  )
}

///|
test "disabled vendor projection preserves all legacy blueprints" {
  let plan = qwen_fused_test_plan()
  let blueprint = qwen_fused_test_blueprint(plan)
  let (_, functions, companions) = collect_paged_functions(blueprint)
  assert_eq(
    paged_vendor_projection_rules(blueprint, functions, companions, 0).length(),
    0,
  )
}
