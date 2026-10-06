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
/// launch rules across attention expansion. A replacement can be bounded only
/// by its own admitted row-domain contract, never the removed operation's rule.
fn remap_decode_split_bucket_rules(
  original : ReadOnlyArray[(@plan.OperationId, Int)],
  expanded : ReadOnlyArray[(@plan.OperationId, Int)],
  contract : PagedGraphLaunchContract,
  row_bounded_replacements? : ReadOnlyArray[
    ((@plan.OperationId, Int), @launch_contract.AotLaunchDimensions),
  ] = [],
) -> FixedArray[PagedBucketStepLaunch] raise DeviceStepError {
  for index, replacement in row_bounded_replacements {
    if original.contains(replacement.0) ||
      expanded.filter(step => step == replacement.0).length() != 1 ||
      row_bounded_replacements[:index].any(item => item.0 == replacement.0) {
      raise InvalidExecutor(GraphBucket)
    }
  }
  FixedArray::makei(expanded.length(), index => {
    let selected = expanded[index]
    for old_index, step in original {
      if step == selected {
        return paged_phase_bucket_rule(contract, DecodeLaunch, old_index)
      }
    }
    for replacement in row_bounded_replacements {
      if replacement.0 == selected {
        return QueryTokenCappedGridX(replacement.1)
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
  let bounded : Array[
    ((@plan.OperationId, Int), @launch_contract.AotLaunchDimensions),
  ] = []
  for companion in resources.attention_partitioned_decode_companions {
    // Only admitted read-only row-indexed ABIs enter this collection. Carry
    // the launch-domain proof through preparation instead of reconstructing
    // it from opaque device handles or inheriting the removed kernel's ABI.
    for
      (step, dimensions) in [
        (companion.partial, companion.partial_dimensions),
        (companion.merge, companion.merge_dimensions),
      ] {
      if dimensions.grid_x() != profile.max_query_rows() {
        raise InvalidExecutor(GraphBucket)
      }
      bounded.push(((step.operation_id, step.function_index), dimensions))
    }
  }
  let rules = remap_decode_split_bucket_rules(
    ReadOnlyArray::from_array(
      original.map(step => (step.operation_id, step.function_index)),
    ),
    ReadOnlyArray::from_array(
      selected.map(step => (step.operation_id, step.function_index)),
    ),
    contract,
    row_bounded_replacements=ReadOnlyArray::from_array(bounded),
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
