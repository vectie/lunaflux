///|
/// Reserve the whole requested family before optional captures: the vendor
/// full graph plus original/vendor prefix and mixed companions. Conservative
/// unused reservations are not spent elsewhere after a capture failure.
fn vendor_capture_family_costs(
  full : Int,
  prefix : Int,
  mixed : Int,
) -> Array[Int] {
  let costs = [full]
  if prefix > 0 && prefix < full {
    costs.append([prefix, prefix])
  }
  if mixed > 0 {
    costs.append([mixed, mixed])
  }
  costs
}

///|
/// Pure partition of the existing structural allowance, never an increase in
/// graph count, node count or an asserted backend-private byte ceiling.
fn partition_vendor_capture_budget(
  budget : @luna_execution_graph_strategy.ExecutionGraphVariantBudget,
  costs : ArrayView[Int],
) -> (
  @luna_execution_graph_strategy.ExecutionGraphVariantBudget,
  @luna_execution_graph_strategy.ExecutionGraphVariantBudget,
) raise DeviceStepError {
  for cost in costs; remaining = budget, nodes = 0L {
    guard remaining.reserve(cost) is Some(next) else {
      raise InvalidExecutor(GraphBucket)
    }
    if nodes + cost.to_int64() > 2147483647L {
      raise InvalidExecutor(GraphBucket)
    }
    continue next, nodes + cost.to_int64()
  } nobreak {
    let requested = @luna_execution_graph_strategy.ExecutionGraphVariantBudget::new(
      costs.length(),
      nodes.to_int(),
    ) catch {
      _ => raise InvalidExecutor(GraphBucket)
    }
    (remaining, requested)
  }
}

///|
fn paged_variant_capture_budget(
  resources : PagedGraphResources,
  bound : Int,
) -> @luna_execution_graph_strategy.ExecutionGraphVariantBudget {
  if bound > 0 && bound == resources.vendor_projection_exact_rows {
    resources.vendor_capture_budget
  } else {
    resources.effect_capture_budget
  }
}

///|
fn update_paged_variant_capture_budget(
  resources : PagedGraphResources,
  bound : Int,
  remaining : @luna_execution_graph_strategy.ExecutionGraphVariantBudget,
) -> Unit {
  if bound > 0 && bound == resources.vendor_projection_exact_rows {
    resources.vendor_capture_budget = remaining
  } else {
    resources.effect_capture_budget = remaining
  }
}
