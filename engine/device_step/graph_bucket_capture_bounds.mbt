///|
fn paged_partitioned_prefill_has_enough_work(
  batch_rows : Int32,
  query_tokens : Int32,
  context_tokens : Int32,
  minimum_context_tokens : Int32,
  query_tile : Int32,
  partition_count : Int32,
) -> Bool {
  if minimum_context_tokens <= 0 || query_tile <= 0 || partition_count <= 1 {
    return false
  }
  let query_tiles = Int32::max(batch_rows, 1 + (query_tokens - 1) / query_tile)
  let context_multiple = if query_tiles <= partition_count {
    2
  } else if query_tiles <= 1073741823 {
    query_tiles * 2
  } else {
    2147483647
  }
  context_tokens >= minimum_context_tokens &&
  context_tokens / minimum_context_tokens >= context_multiple
}

///|
fn paged_partitioned_prefill_frontier_has_enough_work(
  batch_rows : Int32,
  query_tokens : Int32,
  context_tokens : Int32,
  minimum_context_tokens : Int32,
  query_tile : Int32,
  partition_count : Int32,
) -> Bool {
  if minimum_context_tokens <= 0 || query_tile <= 0 || partition_count <= 1 {
    return false
  }
  let query_tiles = Int32::max(batch_rows, 1 + (query_tokens - 1) / query_tile)
  // Deep partitioning supplies missing query parallelism; it must not become
  // easier to select as the query frontier itself grows. Beyond this bound,
  // retain the ordinary prefill/wide path instead of multiplying partials
  // and their merge work on an already populated frontier.
  if query_tiles > partition_count {
    return false
  }
  let frontier_work = if partition_count <= 268435455 {
    partition_count * 8
  } else {
    2147483647
  }
  let context_multiple = if frontier_work == 2147483647 {
    2147483647
  } else {
    Int32::max(2, 1 + (frontier_work - 1) / query_tiles)
  }
  context_tokens >= minimum_context_tokens &&
  context_tokens / minimum_context_tokens >= context_multiple
}

///|
fn paged_decode_capture_bounds(
  profile : @launch_contract.PagedKernelProfile,
) -> ReadOnlyArray[Int32] {
  let maximum = profile.max_query_rows()
  let candidates : FixedArray[Int32] = [1, 8, 16, 32]
  let bounds : Array[Int32] = []
  for candidate in candidates {
    let bound = Int32::min(candidate, maximum)
    if bound < maximum && !bounds.contains(bound) {
      bounds.push(bound)
    }
  }
  if maximum < profile.max_query_tokens() && !bounds.contains(maximum) {
    bounds.push(maximum)
  }
  ReadOnlyArray::from_array(bounds)
}

///|
/// Prefill launch lists are specialized by live query-token capacity rather
/// than replaying the maximum profile envelope. These are execution-graph
/// buckets, not CUDA-specific shapes: a device backend may capture them or
/// execute the same immutable launch list eagerly.
/// Keep the single-token arithmetic domain separate: projection kernels may
/// use one-output-per-subgroup GEMV at one token and tiled GEMM above one.
/// Reusing a multi-token owner's smaller grid can leave output columns unwritten.
fn paged_prefill_capture_bounds(
  profile : @launch_contract.PagedKernelProfile,
) -> ReadOnlyArray[Int32] {
  let maximum = profile.max_query_tokens()
  let candidates : FixedArray[Int32] = [
    1, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096,
  ]
  let bounds : Array[Int32] = []
  for candidate in candidates {
    let bound = Int32::min(candidate, maximum)
    if bound < maximum && !bounds.contains(bound) {
      bounds.push(bound)
    }
  }
  ReadOnlyArray::from_array(bounds)
}
