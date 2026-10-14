///|
test "parallel grouped selection retains selected-order arithmetic" {
  let source = @grouped_routing_cuda_source.grouped_choice_and_selected_weight_f32_source(
    routed_expert_count=288,
    group_count=8,
    group_score_topk=2,
    selected_group_count=4,
    experts_per_token=8,
    normalize=true,
    scaling=2.5,
    maximum_rows=17,
    function_symbol="route_new",
  )
  let text = @utf8.decode(source.parallel_source_bytes())
  assert_true(text.contains("__syncthreads"))
  assert_true(text.contains("denominator=__fadd_rn"))
  assert_true(text.contains("e<ids[lane]"))
  if @env.get_env_var("LUNA_EXPORT_SHAPE_AWARE") == Some("1") {
    let old = @grouped_routing_cuda_source.grouped_choice_and_selected_weight_f32_source(
      routed_expert_count=288,
      group_count=8,
      group_score_topk=2,
      selected_group_count=4,
      experts_per_token=8,
      normalize=true,
      scaling=2.5,
      maximum_rows=17,
      function_symbol="route_old",
    )
    println("LUNA_SOURCE_BEGIN")
    println(@utf8.decode(old.source_bytes()))
    println(text)
    println("LUNA_SOURCE_END")
  }
}
