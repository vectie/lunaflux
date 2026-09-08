///|
test "segmented greedy CUDA preserves legacy ABI and exports map reduce" {
  let source = @luna_cuda_sampling_aot.embedded_greedy_sampling_cuda_source(
    151936, 32,
  )
  assert_true(source.contains("lunaflux_greedy_bf16_rows_v1"))
  assert_true(source.contains("lunaflux_greedy_bf16_partial_v2"))
  assert_true(source.contains("lunaflux_greedy_bf16_merge_v2"))
  let plan = @luna_cuda_sampling_aot.plan_segmented_greedy_cuda(151936, 32)
  assert_eq(plan.partition(), (4096, 38))
  assert_eq(plan.workspace_bytes(), 19456L)
  if @env.get_env_var("LUNA_TEST_EXPORT_GREEDY_SEGMENTS") == Some("1") {
    println("LUNA_SOURCE_BEGIN sampling")
    println(source)
    println("LUNA_SOURCE_END")
  }
}
