///|
fn map_paged_decode_capture_owner(
  table : @luna_execution_graph_strategy.ExecutionGraphBucketTable,
  row_bound : Int32,
  query_token_bound : Int32,
  owner_index : Int32,
  slot_owner_indices : FixedArray[Int32],
) -> Unit raise DeviceStepError {
  for rows = 1; rows <= row_bound; rows = next_paged_graph_bucket_start(rows) {
    for tokens = 1
        tokens <= query_token_bound
        tokens = next_paged_graph_bucket_start(tokens) {
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
  rows : Int32,
  tokens : Int32,
  context : Int32,
  owner_index : Int32,
  slot_owner_indices : FixedArray[Int32],
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
  row_bound : Int32,
  query_token_bound : Int32,
  owner_index : Int32,
  slot_owner_indices : FixedArray[Int32],
) -> Unit raise DeviceStepError {
  let maximum_rows = Int32::min(row_bound, query_token_bound)
  for rows = 1; rows <= maximum_rows; rows = next_paged_graph_bucket_start(rows) {
    for tokens = rows
        tokens <= query_token_bound
        tokens = next_paged_graph_bucket_start(tokens) {
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
  row_bound : Int32,
  query_token_bound : Int32,
  slot_owner_indices : FixedArray[Int32],
) -> Unit raise DeviceStepError {
  for rows = 1; rows <= row_bound; rows = next_paged_graph_bucket_start(rows) {
    for tokens = 1
        tokens <= query_token_bound
        tokens = next_paged_graph_bucket_start(tokens) {
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
