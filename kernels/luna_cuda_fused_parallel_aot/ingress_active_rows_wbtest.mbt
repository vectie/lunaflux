///|
test "ingress producer fold and publication consume the same prefix domain" {
  let candidate = qwen_fused_test_candidate(
    head_dimension=128,
    maximum_tokens=129,
    ingress_policy=qwen_ingress_policy(2, 2, 1, 1, rows=4),
  )
  let physical = qwen_refine_physical(
    candidate.tile_compilation(),
    candidate.ingress_policy(),
  )
  let domain = physical.pipeline.row_domain
  assert_eq(domain.geometry(), (4, 16))
  for
    (remaining, expected) in [
      (1, 1),
      (16, 1),
      (17, 2),
      (32, 2),
      (33, 3),
      (48, 3),
      (49, 4),
      (64, 4),
      (65, 4),
    ] {
    assert_eq(domain.active_groups(remaining), expected)
  }
  let source = @utf8.decode(candidate.source_bytes())
  assert_true(source.contains("row_fragment >= lf_active_row_fragments"))
  assert_true(source.contains("lf_active_row_fragments == 1"))
  assert_true(source.contains("lf_active_row_fragments == 4"))
  assert_true(source.contains("row_fragment < ingress_active_fragments"))
  let vectors_per_fragment = 16 *
    physical.pipeline.schedule.transfer_width() /
    8
  assert_true(
    source.contains("ingress_active_fragments * \{vectors_per_fragment}"),
  )
  // Zero-fill remains for lanes of a partially valid final instruction group;
  // block-wide publication/retirement barriers remain outside row predicates.
  assert_true(source.contains("valid ? 16 : 0"))
  assert_true(source.contains("joint-head-products-v1"))
}
