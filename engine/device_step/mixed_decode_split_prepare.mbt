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
  phase? : PagedBucketLaunchPhase = BaselineLaunch,
  capture_budget_reserved? : Bool = false,
) -> FixedArray[Int] raise DeviceStepError {
  let slots = FixedArray::make(table.bucket_count(), -1)
  guard candidates is Some(ids) else { return slots }
  if ids.length() != slots.length() || baseline.length() != slots.length() {
    raise InvalidExecutor(GraphBucket)
  }
  let prepared : Array[(Int, Int, Int, Int)] = []
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
    let rows = bucket.batch_rows()
    if find_mixed_attention_item(prepared[:], item => {
        item.0 == owner && item.1 == bound && item.2 == rows
      })
      is Some(found) {
      slots[slot] = found.3
      continue
    }
    let launches = paged_ordered_launches(
      steps,
      resources,
      phase~,
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
        capture_budget_reserved~,
        decode_row_bound=Some(rows),
      )
      is Some(mixed) else {
      // A measured winner must actually be executable: missing read-only
      // companions or a capture-budget failure cannot be relabeled as a win.
      raise InvalidExecutor(GraphBucket)
    }
    slots[slot] = mixed
    prepared.push((owner, bound, rows, mixed))
  }
  slots
}

///|
/// Pure startup reservation: measured mandatory captures precede optional
/// variants. Context buckets sharing the same row/query domain share a graph.
/// Neither the graph/node allowance nor deployment byte ceilings are raised.
fn reserve_measured_mixed_capture_budget(
  budget : @luna_execution_graph_strategy.ExecutionGraphVariantBudget,
  table : @luna_execution_graph_strategy.ExecutionGraphBucketTable,
  candidates : ReadOnlyArray[Int]?,
  nodes : Int,
) -> @luna_execution_graph_strategy.ExecutionGraphVariantBudget raise DeviceStepError {
  guard candidates is Some(ids) else { return budget }
  if ids.length() != table.bucket_count() {
    raise InvalidExecutor(GraphBucket)
  }
  let domains : Array[(Int, Int)] = []
  let mut remaining = budget
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
    let domain = (bucket.batch_rows(), bucket.query_tokens())
    if domains.contains(domain) {
      continue
    }
    remaining = remaining
      .reserve(nodes)
      .unwrap_or_else(() => raise InvalidExecutor(GraphBucket))
    domains.push(domain)
  }
  remaining
}
