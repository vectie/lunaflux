///|
fn paged_graph_maximum_context_tokens() -> Int {
  1048576
}

///|
fn new_paged_graph_bucket_table(
  profile : @launch_contract.PagedKernelProfile,
) -> @luna_execution_graph_strategy.ExecutionGraphBucketTable {
  @luna_execution_graph_strategy.build_execution_graph_bucket_table(
    profile.max_query_rows(),
    profile.max_query_tokens(),
    paged_graph_maximum_context_tokens(),
    65536,
  ) catch {
    _ => abort("validated paged profile rejected by graph buckets")
  }
}

///|
fn paged_graph_phase(
  prefill : Int,
  decode : Int,
) -> @luna_execution_graph_strategy.ExecutionGraphPhase {
  if prefill > 0 && decode == 0 {
    PrefillGraph
  } else if prefill == 0 && decode > 0 {
    DecodeGraph
  } else {
    MixedGraph
  }
}

///|
fn paged_maximum_graph_slots(
  table : @luna_execution_graph_strategy.ExecutionGraphBucketTable,
  profile : @launch_contract.PagedKernelProfile,
) -> FixedArray[Int] raise DeviceStepError {
  let slots = FixedArray::make(3, -1)
  let phases : FixedArray[@luna_execution_graph_strategy.ExecutionGraphPhase] = [
    PrefillGraph,
    DecodeGraph,
    MixedGraph,
  ]
  for index, phase in phases {
    let query_tokens = if phase == DecodeGraph {
      profile.max_query_rows()
    } else {
      profile.max_query_tokens()
    }
    let shape = @luna_execution_graph_strategy.ExecutionGraphShape::new(
      phase~,
      batch_rows=profile.max_query_rows(),
      query_tokens~,
      context_tokens=paged_graph_maximum_context_tokens(),
    ) catch {
      _ => raise InvalidExecutor(GraphBucket)
    }
    slots[index] = table.select(shape).slot() catch {
        _ => raise InvalidExecutor(GraphBucket)
      }
  }
  slots
}

///|
fn PagedOrderedExecutorDispatch::select(
  self : PagedOrderedExecutorDispatch,
  bucket : @luna_execution_graph_strategy.ExecutionGraphBucket,
) -> Int raise DeviceStepError {
  let (_, _, context_tokens) = bucket.upper_bounds()
  if self.prefill_split_owner_index >= 0 &&
    (bucket.phase() == PrefillGraph || bucket.phase() == MixedGraph) &&
    context_tokens >= self.prefill_split_minimum_context_tokens {
    return self.prefill_split_owner_index
  }
  if paged_attention_phase_route(
      bucket.phase(),
      self.decode_split_owner_index >= 0,
    ) ==
    DecodeSplitAttentionPhase &&
    context_tokens >= self.decode_split_minimum_context_tokens {
    return self.decode_split_owner_index
  }
  let slot = bucket.slot()
  if slot >= 0 && slot < self.slot_owner_indices.length() {
    let owner = self.slot_owner_indices[slot]
    if owner >= 0 {
      return owner
    }
  }
  if bucket.phase() == DecodeGraph && self.decode_owner_index >= 0 {
    return self.decode_owner_index
  }
  if self.fallback_owner_index >= 0 {
    self.fallback_owner_index
  } else {
    raise InvalidExecutor(GraphBucket)
  }
}

///|
fn paged_decode_capture_bounds(
  profile : @launch_contract.PagedKernelProfile,
) -> ReadOnlyArray[Int] {
  let maximum = profile.max_query_rows()
  let candidates : FixedArray[Int] = [1, 8, 16, 32]
  let bounds : Array[Int] = []
  for candidate in candidates {
    let bound = Int::min(candidate, maximum)
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
fn paged_prefill_capture_bounds(
  profile : @launch_contract.PagedKernelProfile,
) -> ReadOnlyArray[Int] {
  let maximum = profile.max_query_tokens()
  let candidates : FixedArray[Int] = [
    8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096,
  ]
  let bounds : Array[Int] = []
  for candidate in candidates {
    let bound = Int::min(candidate, maximum)
    if bound < maximum && !bounds.contains(bound) {
      bounds.push(bound)
    }
  }
  ReadOnlyArray::from_array(bounds)
}

///|
fn map_paged_decode_capture_owner(
  table : @luna_execution_graph_strategy.ExecutionGraphBucketTable,
  row_bound : Int,
  query_token_bound : Int,
  owner_index : Int,
  slot_owner_indices : FixedArray[Int],
) -> Unit raise DeviceStepError {
  for rows in 1..<=row_bound {
    for tokens in 1..<=query_token_bound {
      let mut context = 1
      while context <= paged_graph_maximum_context_tokens() {
        let slot = table
          .select(
            @luna_execution_graph_strategy.ExecutionGraphShape::new(
              phase=DecodeGraph,
              batch_rows=rows,
              query_tokens=tokens,
              context_tokens=context,
            ) catch {
              _ => raise InvalidExecutor(GraphBucket)
            },
          )
          .slot() catch {
            _ => raise InvalidExecutor(GraphBucket)
          }
        if slot_owner_indices[slot] < 0 {
          slot_owner_indices[slot] = owner_index
        }
        context *= 2
      }
    }
  }
}

///|
fn map_paged_graph_owner_slot(
  table : @luna_execution_graph_strategy.ExecutionGraphBucketTable,
  phase : @luna_execution_graph_strategy.ExecutionGraphPhase,
  rows : Int,
  tokens : Int,
  context : Int,
  owner_index : Int,
  slot_owner_indices : FixedArray[Int],
) -> Unit raise DeviceStepError {
  let slot = table
    .select(
      @luna_execution_graph_strategy.ExecutionGraphShape::new(
        phase~,
        batch_rows=rows,
        query_tokens=tokens,
        context_tokens=context,
      ) catch {
        _ => raise InvalidExecutor(GraphBucket)
      },
    )
    .slot() catch {
      _ => raise InvalidExecutor(GraphBucket)
    }
  if slot_owner_indices[slot] < 0 {
    slot_owner_indices[slot] = owner_index
  }
}

///|
/// Maps every valid prefill or continuous-batching mixed shape covered by one
/// query-token capacity to its smallest already-built owner. Every row
/// contributes at least one query token. A mixed shape additionally requires
/// at least two rows because both phases must be present. Both phases consume
/// the same immutable baseline launch list and differ only in descriptor
/// counts, so no backend-specific graph concept enters this mapping.
fn map_paged_prefill_capture_owner(
  table : @luna_execution_graph_strategy.ExecutionGraphBucketTable,
  row_bound : Int,
  query_token_bound : Int,
  owner_index : Int,
  slot_owner_indices : FixedArray[Int],
) -> Unit raise DeviceStepError {
  let maximum_rows = Int::min(row_bound, query_token_bound)
  for rows in 1..<=maximum_rows {
    for tokens in rows..<=query_token_bound {
      let mut context = 1
      while context <= paged_graph_maximum_context_tokens() {
        map_paged_graph_owner_slot(
          table,
          PrefillGraph,
          rows,
          tokens,
          context,
          owner_index,
          slot_owner_indices,
        )
        if rows >= 2 {
          map_paged_graph_owner_slot(
            table,
            MixedGraph,
            rows,
            tokens,
            context,
            owner_index,
            slot_owner_indices,
          )
        }
        context *= 2
      }
    }
  }
}

///|
/// Removes only decode-slot owners before a phase-specific decode launch list
/// is materialized. Prefill and mixed owners remain intact. The following
/// ascending-bound mapping then preserves the smallest compatible owner for
/// every decode bucket.
fn clear_paged_decode_capture_owners(
  table : @luna_execution_graph_strategy.ExecutionGraphBucketTable,
  row_bound : Int,
  query_token_bound : Int,
  slot_owner_indices : FixedArray[Int],
) -> Unit raise DeviceStepError {
  for rows in 1..<=row_bound {
    for tokens in 1..<=query_token_bound {
      let mut context = 1
      while context <= paged_graph_maximum_context_tokens() {
        let slot = table
          .select(
            @luna_execution_graph_strategy.ExecutionGraphShape::new(
              phase=DecodeGraph,
              batch_rows=rows,
              query_tokens=tokens,
              context_tokens=context,
            ) catch {
              _ => raise InvalidExecutor(GraphBucket)
            },
          )
          .slot() catch {
            _ => raise InvalidExecutor(GraphBucket)
          }
        slot_owner_indices[slot] = -1
        context *= 2
      }
    }
  }
}

///|
fn select_submitted_graph_bucket(
  table : @luna_execution_graph_strategy.ExecutionGraphBucketTable,
  plan : @worker_protocol.SubmittedSchedulePlan,
) -> @luna_execution_graph_strategy.ExecutionGraphBucket raise DeviceStepError {
  let (prefill, decode, rows, tokens) = (
    plan.prefill_count(),
    plan.decode_count(),
    plan.row_count(),
    plan.token_count(),
  ) catch {
    error => raise Protocol(error)
  }
  let mut context_tokens = 1
  for index in 0..<prefill {
    let end = try {
      let row = plan.prefill_row(index)
      row.sequence_token_start() + row.tokens().length()
    } catch {
      error => raise Protocol(error)
    }
    if end > context_tokens {
      context_tokens = end
    }
  }
  for index in 0..<decode {
    let end = try plan.decode_row(index).sequence_token_position() + 1 catch {
      error => raise Protocol(error)
    }
    if end > context_tokens {
      context_tokens = end
    }
  }
  let shape = @luna_execution_graph_strategy.ExecutionGraphShape::new(
    phase=paged_graph_phase(prefill, decode),
    batch_rows=rows,
    query_tokens=tokens,
    context_tokens~,
  ) catch {
    _ => raise InvalidExecutor(GraphBucket)
  }
  table.select(shape) catch {
    _ => raise InvalidExecutor(GraphBucket)
  }
}

///|
fn select_frame_graph_bucket(
  table : @luna_execution_graph_strategy.ExecutionGraphBucketTable,
  frame : @worker_wire.ValidatedPlanFrame,
) -> @luna_execution_graph_strategy.ExecutionGraphBucket raise DeviceStepError {
  let (prefill, decode, tokens) = (
    frame.prefill_count(),
    frame.decode_count(),
    frame.token_count(),
  ) catch {
    error => raise Wire(error)
  }
  let rows = prefill + decode
  let mut context_tokens = 1
  for index in 0..<prefill {
    let end = try {
      let row = frame.prefill_row(index)
      row.sequence_token_start() + row.token_length()
    } catch {
      error => raise Wire(error)
    }
    if end > context_tokens {
      context_tokens = end
    }
  }
  for index in 0..<decode {
    let end = try frame.decode_row(index).sequence_token_position() + 1 catch {
      error => raise Wire(error)
    }
    if end > context_tokens {
      context_tokens = end
    }
  }
  let shape = @luna_execution_graph_strategy.ExecutionGraphShape::new(
    phase=paged_graph_phase(prefill, decode),
    batch_rows=rows,
    query_tokens=tokens,
    context_tokens~,
  ) catch {
    _ => raise InvalidExecutor(GraphBucket)
  }
  table.select(shape) catch {
    _ => raise InvalidExecutor(GraphBucket)
  }
}
