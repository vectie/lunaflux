///|
fn paged_ordered_device_policy(
  policy : PagedGraphExecutionPolicy,
) -> @device.OrderedKernelExecutionPolicy {
  match policy {
    PagedOrderedEagerOnly => OrderedKernelEagerOnly
    PagedCapturedRequired => OrderedKernelCaptureRequired
    PagedCapturedWithEagerFallback => OrderedKernelCaptureWithEagerFallback
  }
}

///|
/// Pure startup transformation: unchanged operations retain their bounded
/// launch rules across attention expansion. New partial/reduce kernels keep
/// their own exact launch contracts; a rule from another ABI is never reused.
fn remap_decode_split_bucket_rules(
  original : ReadOnlyArray[(@plan.OperationId, Int)],
  expanded : ReadOnlyArray[(@plan.OperationId, Int)],
  contract : PagedGraphLaunchContract,
) -> FixedArray[PagedBucketStepLaunch] {
  FixedArray::makei(expanded.length(), index => {
    let selected = expanded[index]
    for old_index, step in original {
      if step == selected {
        return paged_phase_bucket_rule(contract, DecodeLaunch, old_index)
      }
    }
    FixedBucketLaunch
  })
}

///|
fn prepare_decode_split_bucket_owners(
  context : @device.Context,
  table : @luna_execution_graph_strategy.ExecutionGraphBucketTable,
  profile : @launch_contract.PagedKernelProfile,
  original : ReadOnlyArray[PreparedPagedGraphStep],
  selected : ReadOnlyArray[PreparedPagedGraphStep],
  operation_ids : ReadOnlyArray[@plan.OperationId],
  resources : PagedGraphResources,
  policy : PagedGraphExecutionPolicy,
  slots : FixedArray[Int],
) -> Unit raise DeviceStepError {
  guard resources.graph_launch_contract is Some(contract) else { return }
  let rules = remap_decode_split_bucket_rules(
    ReadOnlyArray::from_array(
      original.map(step => (step.operation_id, step.function_index)),
    ),
    ReadOnlyArray::from_array(
      selected.map(step => (step.operation_id, step.function_index)),
    ),
    contract,
  )
  for bound in paged_decode_capture_bounds(profile) {
    let launches = paged_ordered_launches(
      selected,
      resources,
      query_token_bound=Some(bound),
      phase=DecodeLaunch,
      launch_rules=Some(rules),
    )
    let owner = prepare_one_paged_ordered_executor(
      context,
      launches[:],
      operation_ids,
      resources,
      paged_ordered_device_policy(policy),
      source_steps=selected,
    )
    map_paged_decode_capture_owner(table, bound, bound, owner, slots)
  }
}
