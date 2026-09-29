///|
test "packed ingress physical fixtures cover segment boundaries and head tails" {
  for (dimension, heads) in [(64, 1), (64, 2), (64, 4), (128, 1), (128, 2)] {
    let (query_heads, kv_heads) = if dimension == 64 { (6, 3) } else { (1, 1) }
    let candidate = qwen_fused_test_candidate(
      head_dimension=dimension,
      maximum_tokens=129,
      query_heads~,
      kv_heads~,
      ingress_policy=qwen_ingress_policy(4, 2, 1, 1, heads~),
    )
    let production = lower_fused_qwen_qkv_qknorm_rope_paged_kv_write_production_candidate(
      candidate,
      fused_test_limits(),
    )
    assert_eq(
      candidate.launch_dimensions().grid_y(),
      (query_heads + 2 * kv_heads - 1) / heads + 1,
    )
    if @env.get_env_var("LUNA_TEST_EXPORT_COUNTER_REPAIR") == Some("1") {
      println("LUNA_SOURCE_BEGIN ingress-pack-d\{dimension}-h\{heads}")
      println(@utf8.decode(production.source_bytes()))
      println("LUNA_SOURCE_END")
    }
  }
}
