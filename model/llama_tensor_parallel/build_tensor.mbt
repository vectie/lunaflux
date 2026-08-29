///|
fn llama_tp_append_tensor(
  output : Array[@tensor_parallel_plan.TensorRankPlacement],
  bindings : @llama_weights.LlamaWeightBindings,
  topology : @tensor_parallel_plan.TensorParallelTopology,
  reference : Int,
  placement : @tensor_parallel_plan.TensorPlacement,
  rows : Int64,
  columns : Int64,
  alignment : Int64,
  arena_end : Ref[Int64],
  materialized : Ref[Int64],
  max_arena_bytes : Int64,
) -> Unit raise LlamaTensorParallelError {
  let tensor = llama_tp_require_binding_shape(
    bindings, reference, rows, columns,
  )
  let world = topology.world_size().to_int64()
  let rank = topology.rank().to_int64()
  let effective_placement = if topology.world_size() == 1 {
    @tensor_parallel_plan.TensorPlacement::Replicated
  } else {
    placement
  }
  let (row_start, row_count, column_start, column_count) = match
    effective_placement {
    Replicated => (0L, rows, 0L, columns)
    ShardRows => {
      if rows % world != 0L {
        raise UnsupportedBinding(reference~)
      }
      let count = rows / world
      (rank * count, count, 0L, columns)
    }
    ShardColumns => {
      if columns % world != 0L {
        raise UnsupportedBinding(reference~)
      }
      let count = columns / world
      (0L, rows, rank * count, count)
    }
  }
  let local_elements = llama_tp_checked_mul(row_count, column_count, reference)
  let local_bytes = llama_tp_checked_mul(local_elements, 2L, reference)
  let full_bytes = llama_tp_checked_mul(
    llama_tp_checked_mul(rows, columns, reference),
    2L,
    reference,
  )
  if topology.world_size() > 1 &&
    effective_placement != Replicated &&
    local_bytes >= full_bytes {
    raise UnsupportedBinding(reference~)
  }
  let destination = llama_tp_checked_align(arena_end.val, alignment, reference)
  let end = llama_tp_checked_add(destination, local_bytes, reference)
  if end > max_arena_bytes {
    raise ArenaTooLarge
  }
  let full_row_bytes = llama_tp_checked_mul(columns, 2L, reference)
  let local_row_bytes = llama_tp_checked_mul(column_count, 2L, reference)
  let (source_base, source_stride, destination_stride, segment_bytes, segments) = match
    effective_placement {
    Replicated => (tensor.data_start(), 0L, 0L, local_bytes, 1L)
    ShardRows =>
      (
        llama_tp_checked_add(
          tensor.data_start(),
          llama_tp_checked_mul(row_start, full_row_bytes, reference),
          reference,
        ),
        0L,
        0L,
        local_bytes,
        1L,
      )
    ShardColumns =>
      (
        llama_tp_checked_add(
          tensor.data_start(),
          llama_tp_checked_mul(column_start, 2L, reference),
          reference,
        ),
        full_row_bytes,
        local_row_bytes,
        local_row_bytes,
        rows,
      )
  }
  let tensor_reference = @plan.TensorRef::new(reference) catch {
    _ => raise UnsupportedBinding(reference~)
  }
  let region = @tensor_parallel_plan.TensorArenaRegion::new(
    offset_bytes=destination,
    byte_length=local_bytes,
    alignment_bytes=alignment,
    max_arena_bytes~,
  ) catch {
    InvalidTensorParallelPlan(..) as error => raise InvalidPhysicalPlan(error)
  }
  let extent = @tensor_parallel_plan.TensorShardExtent::new(
    full_rows=rows,
    full_columns=columns,
    row_start~,
    row_count~,
    column_start~,
    column_count~,
    byte_length=local_bytes,
  ) catch {
    InvalidTensorParallelPlan(..) as error => raise InvalidPhysicalPlan(error)
  }
  let recipe = @tensor_parallel_plan.TensorTransferRecipe::new(
    reference=tensor_reference,
    source_base_bytes=source_base,
    source_stride_bytes=source_stride,
    destination_base_bytes=destination,
    destination_stride_bytes=destination_stride,
    segment_bytes~,
    segment_count=segments,
    destination_region=region,
  ) catch {
    InvalidTensorParallelPlan(..) as error => raise InvalidPhysicalPlan(error)
  }
  output.push(
    @tensor_parallel_plan.TensorRankPlacement::new(
      reference=tensor_reference,
      placement=effective_placement,
      extent~,
      region~,
      transfer_recipe=recipe,
    ) catch {
      InvalidTensorParallelPlan(..) as error => raise InvalidPhysicalPlan(error)
    },
  )
  arena_end.val = end
  materialized.val = llama_tp_checked_add(
    materialized.val,
    local_bytes,
    reference,
  )
}
