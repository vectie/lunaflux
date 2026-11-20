///|
fn paged_graph_maximum_context_tokens() -> Int32 {
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
  prefill : Int32,
  decode : Int32,
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
) -> FixedArray[Int32] raise DeviceStepError {
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
) -> Int32 raise DeviceStepError {
  if bucket.slot() < self.measured_owner_indices.length() {
    let owner = self.measured_owner_indices[bucket.slot()]
    if owner >= 0 {
      return owner
    }
  }
  let batch_rows = bucket.batch_rows()
  let query_tokens = bucket.query_tokens()
  let context_tokens = bucket.context_tokens()
  let prefill_phase = bucket.phase() == PrefillGraph ||
    bucket.phase() == MixedGraph
  if self.deep_prefill_split_owner_index >= 0 &&
    prefill_phase &&
    paged_partitioned_prefill_frontier_has_enough_work(
      batch_rows,
      query_tokens,
      context_tokens,
      self.deep_prefill_split_minimum_context_tokens,
      self.deep_prefill_split_query_tile,
      self.deep_prefill_split_partition_count,
    ) {
    return self.deep_prefill_split_owner_index
  }
  if self.prefill_split_owner_index >= 0 &&
    prefill_phase &&
    paged_partitioned_prefill_has_enough_work(
      batch_rows,
      query_tokens,
      context_tokens,
      self.prefill_split_minimum_context_tokens,
      self.prefill_split_query_tile,
      self.prefill_split_partition_count,
    ) {
    let slot = bucket.slot()
    if slot >= 0 && slot < self.prefill_split_slot_owner_indices.length() {
      let owner = self.prefill_split_slot_owner_indices[slot]
      if owner >= 0 {
        return owner
      }
    }
    return self.prefill_split_owner_index
  }
  if paged_attention_phase_route(
      bucket.phase(),
      self.decode_split_owner_index >= 0,
    ) ==
    DecodeSplitAttentionPhase &&
    context_tokens >= self.decode_split_minimum_context_tokens &&
    batch_rows <= self.decode_split_maximum_batch_rows {
    let slot = bucket.slot()
    if slot >= 0 && slot < self.decode_split_slot_owner_indices.length() {
      let owner = self.decode_split_slot_owner_indices[slot]
      if owner >= 0 {
        return owner
      }
    }
    return self.decode_split_owner_index
  }
  if self.prefill_wide_owner_index >= 0 &&
    (bucket.phase() == PrefillGraph || bucket.phase() == MixedGraph) &&
    query_tokens >= self.prefill_wide_minimum_query_tokens {
    let slot = bucket.slot()
    if slot >= 0 && slot < self.prefill_wide_slot_owner_indices.length() {
      let owner = self.prefill_wide_slot_owner_indices[slot]
      if owner >= 0 {
        return owner
      }
    }
    return self.prefill_wide_owner_index
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
