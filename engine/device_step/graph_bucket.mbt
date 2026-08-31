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
fn paged_graph_phase_index(
  phase : @luna_execution_graph_strategy.ExecutionGraphPhase,
) -> Int {
  match phase {
    PrefillGraph => 0
    DecodeGraph => 1
    MixedGraph => 2
  }
}

///|
fn paged_exact_maximum_graph_slots(
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
    let shape = @luna_execution_graph_strategy.ExecutionGraphShape::new(
      phase~,
      batch_rows=profile.max_query_rows(),
      query_tokens=profile.max_query_tokens(),
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
  let phase = paged_graph_phase_index(bucket.phase())
  if bucket.slot() == self.exact_slots[phase] {
    let exact = self.exact_slot_owner_indices[phase]
    if exact >= 0 {
      return exact
    }
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
