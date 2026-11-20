///|
fn reserve_paged_vendor_capture_budget(
  steps : ReadOnlyArray[PreparedPagedGraphStep],
  resources : PagedGraphResources,
  graph_policy : PagedGraphExecutionPolicy,
  profile : @launch_contract.PagedKernelProfile,
  prefill_split_steps : ReadOnlyArray[PreparedPagedGraphStep]?,
  prefill_wide_steps : ReadOnlyArray[PreparedPagedGraphStep]?,
  prefill_wide_minimum_query_tokens : Int32,
) -> Unit raise DeviceStepError {
  if resources.vendor_projection_exact_rows > 0 {
    let rows = resources.vendor_projection_exact_rows
    if graph_policy == PagedOrderedEagerOnly ||
      resources.graph_launch_contract is None ||
      (
        rows != profile.max_query_tokens() &&
        !paged_prefill_capture_bounds(profile).contains(rows)
      ) {
      raise InvalidExecutor(GraphBucket)
    }
    let families = [steps]
    if prefill_split_steps is Some(split) {
      families.push(split)
    }
    if prefill_wide_steps is Some(wide) &&
      (
        rows == profile.max_query_tokens() ||
        rows >= prefill_wide_minimum_query_tokens
      ) {
      families.push(wide)
    }
    let costs : Array[Int32] = []
    for family in families {
      let operations = ReadOnlyArray::from_array(
        family.map(step => step.operation_id),
      )
      let prefix = effect_only_launch_count(
        operations,
        resources.effect_operation_prefix,
        preparation_prefix=resources.effect_preparation_prefix,
      )
      let mixed = mixed_attention_steps(
        family,
        resources,
        partitioned_decode=false,
      )
      costs.append(
        vendor_capture_family_costs(
          family.length(),
          prefix,
          match mixed {
            Some(value) => value.length()
            None => 0
          },
        ),
      )
    }
    let (optional, requested) = partition_vendor_capture_budget(
      resources.effect_capture_budget,
      costs[:],
    )
    resources.effect_capture_budget = optional
    resources.vendor_capture_budget = requested
  }
}
