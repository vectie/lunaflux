///|
test "requested vendor capture families cannot be starved by earlier optional graphs" {
  let costs = vendor_capture_family_costs(202, 194, 230)
  assert_eq(costs, [202, 194, 194, 230, 230])
  let budget = @luna_execution_graph_strategy.ExecutionGraphVariantBudget::new(
    6, 1100,
  )
  let (optional, requested) = partition_vendor_capture_budget(budget, costs[:])
  assert_eq(optional.reserve(51), None)
  assert_eq(optional.reserve(50).unwrap().reserve(1), None)
  // Optional preparation cannot mutate the immutable reserved allowance.
  let (empty, _) = partition_vendor_capture_budget(requested, costs[:])
  assert_eq(empty.reserve(1), None)
  let resources = empty_paged_resources()
  resources.vendor_projection_exact_rows = 2048
  resources.vendor_capture_budget = requested
  resources.effect_capture_budget = optional
  assert_eq(paged_variant_capture_budget(resources, 2048), requested)
  assert_eq(paged_variant_capture_budget(resources, 1024), optional)
  assert_eq(paged_variant_capture_budget(resources, 0), optional)
}

///|
test "requested vendor reservation fails closed without widening graph or node limits" {
  assert_eq(vendor_capture_family_costs(202, 202, 0), [202])
  for
    budget in [
      @luna_execution_graph_strategy.ExecutionGraphVariantBudget::new(4, 2000),
      @luna_execution_graph_strategy.ExecutionGraphVariantBudget::new(5, 1049),
      effect_capture_budget(GraphMemoryBounded),
    ] {
    try
      partition_vendor_capture_budget(
        budget,
        vendor_capture_family_costs(202, 194, 230)[:],
      )
    catch {
      InvalidExecutor(GraphBucket) => continue
      _ => fail("unexpected error")
    } noraise {
      _ => fail("overcommitted graph family")
    }
  }
}
