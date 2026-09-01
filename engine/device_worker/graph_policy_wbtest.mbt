///|
test "BF16 production graph policy derives eager and explicit fallback exactly" {
  assert_eq(
    select_authenticated_bf16_graph_policy(false, false),
    PagedOrderedEagerOnly,
  )
  assert_eq(
    select_authenticated_bf16_graph_policy(false, true),
    PagedCapturedWithEagerFallback,
  )
  assert_eq(
    select_authenticated_bf16_graph_policy(true, false),
    PagedOrderedEagerOnly,
  )
  assert_eq(
    select_authenticated_bf16_graph_policy(true, true),
    PagedCapturedWithEagerFallback,
  )
}

///|
test "BF16 production graph policy does not fabricate capture-required authority" {
  for has_authorization in [false, true] {
    for capture_safe in [false, true] {
      assert_false(
        select_authenticated_bf16_graph_policy(has_authorization, capture_safe) ==
        PagedCapturedRequired,
      )
    }
  }
}

///|
test "BF16 production graph policy does not gate stable AOT capture on external approval" {
  assert_eq(
    select_authenticated_bf16_graph_policy(false, true),
    PagedCapturedWithEagerFallback,
  )
  assert_eq(
    select_authenticated_bf16_graph_policy(true, true),
    PagedCapturedWithEagerFallback,
  )
}
