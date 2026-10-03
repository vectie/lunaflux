///|
/// Instantiate only measured mixed winners. Ordinary mixed graphs remain
/// available for baseline choices, and unmeasured buckets retain their policy.
/// New variants reuse the existing read-only partial/merge functions and
/// scratch, and are charged to the startup graph-memory budget.
fn prepare_measured_mixed_decode_split_owners(
  context : @device.Context,
  steps : ReadOnlyArray[PreparedPagedGraphStep],
  resources : PagedGraphResources,
  table : @luna_execution_graph_strategy.ExecutionGraphBucketTable,
  candidates : ReadOnlyArray[Int]?,
  baseline : FixedArray[Int],
  fallback : Int,
) -> FixedArray[Int] raise DeviceStepError {
  let slots = FixedArray::make(table.bucket_count(), -1)
  guard candidates is Some(ids) else { return slots }
  if ids.length() != slots.length() || baseline.length() != slots.length() {
    raise InvalidExecutor(GraphBucket)
  }
  let prepared : Array[(Int, Int, Int)] = []
  for slot, candidate in ids {
    if candidate != 7 {
      continue
    }
    let bucket = table.bucket_at(slot) catch {
      _ => raise InvalidExecutor(GraphBucket)
    }
    if bucket.phase() != MixedGraph {
      raise InvalidExecutor(GraphBucket)
    }
    let owner = if baseline[slot] >= 0 { baseline[slot] } else { fallback }
    let bound = bucket.query_tokens()
    if find_mixed_attention_item(prepared[:], item => {
        item.0 == owner && item.1 == bound
      })
      is Some(found) {
      slots[slot] = found.2
      continue
    }
    let launches = paged_ordered_launches(
      steps,
      resources,
      query_token_bound=if resources.graph_launch_contract is Some(_) {
        Some(bound)
      } else {
        None
      },
    )
    guard prepare_mixed_attention_variant(
        context,
        steps,
        launches[:],
        resources,
        owner,
        partitioned_decode=true,
      )
      is Some(mixed) else {
      // A measured winner must actually be executable: missing read-only
      // companions or a capture-budget failure cannot be relabeled as a win.
      raise InvalidExecutor(GraphBucket)
    }
    slots[slot] = mixed
    prepared.push((owner, bound, mixed))
  }
  slots
}
