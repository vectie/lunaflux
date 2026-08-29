///|
const MAX_INT64 : Int64 = 9223372036854775807L

///|
fn tensor_reference(
  index : Int,
) -> @plan.TensorRef raise TensorParallelDevicePlanError {
  @plan.TensorRef::new(index) catch {
    _ => raise invalid(TensorReference, index)
  }
}

///|
fn checked_bf16_bytes(
  rows : Int64,
  columns : Int64,
  index : Int,
) -> Int64 raise TensorParallelDevicePlanError {
  if rows <= 0L ||
    columns <= 0L ||
    rows > MAX_INT64 / columns ||
    rows * columns > MAX_INT64 / 2L {
    raise invalid(TensorExtent, index)
  }
  rows * columns * 2L
}

///|
fn validate_operation_layout(
  model_plan : @plan.ModelPlan,
  rank_plan : @tensor_parallel_plan.TensorParallelRankPlan,
) -> ReadOnlyArray[@tensor_parallel_plan.TensorParallelOperationPlan] raise TensorParallelDevicePlanError {
  let semantic = model_plan.operations()
  if semantic.is_empty() || rank_plan.operation_count() != semantic.length() {
    raise invalid(OperationLayout, rank_plan.operation_count())
  }
  let referenced = FixedArray::make(rank_plan.tensor_count(), false)
  let output : Array[@tensor_parallel_plan.TensorParallelOperationPlan] = []
  for index, operation in semantic {
    guard rank_plan.operation(operation.id()) is Some(geometry) else {
      raise invalid(OperationLayout, index)
    }
    if geometry.operation().as_int() != index ||
      geometry.kind() != operation.kind() ||
      geometry.capability() != operation.required_capability() ||
      geometry.input_width() <= 0 ||
      geometry.output_width() <= 0 {
      raise invalid(OperationLayout, index)
    }
    for input in operation.inputs() {
      match input {
        Tensor(reference) => {
          let tensor_index = reference.as_int()
          if tensor_index < 0 || tensor_index >= referenced.length() {
            raise invalid(TensorReference, tensor_index)
          }
          referenced[tensor_index] = true
        }
        _ => ()
      }
    }
    output.push(geometry)
  }
  for index, used in referenced {
    if !used {
      raise invalid(TensorReference, index)
    }
  }
  ReadOnlyArray::from_array(output)
}

///|
fn require_matrix(
  rank_plan : @tensor_parallel_plan.TensorParallelRankPlan,
  reference : Int,
) -> TensorParallelMatrixView raise TensorParallelDevicePlanError {
  let tensor = tensor_reference(reference)
  guard rank_plan.tensor(tensor) is Some(placement) else {
    raise invalid(TensorReference, reference)
  }
  if placement.reference() != tensor {
    raise invalid(TensorReference, reference)
  }
  let extent = placement.extent()
  let region = placement.region()
  let local_bytes = checked_bf16_bytes(
    extent.row_count(),
    extent.column_count(),
    reference,
  )
  if extent.byte_length() != local_bytes ||
    region.byte_length() != local_bytes ||
    region.offset_bytes() < 0L ||
    region.offset_bytes() % rank_plan.alignment_bytes() != 0L ||
    region.offset_bytes() > MAX_INT64 - region.byte_length() ||
    region.end_offset_bytes() > rank_plan.arena_bytes() {
    raise invalid(TensorRegion, reference)
  }
  match placement.placement() {
    Replicated => ()
    ShardRows =>
      if extent.row_count() >= extent.full_rows() {
        raise invalid(FullShardedTensor, reference)
      }
    ShardColumns =>
      if extent.column_count() >= extent.full_columns() {
        raise invalid(FullShardedTensor, reference)
      }
  }
  {
    reference: tensor,
    placement: placement.placement(),
    full_rows: extent.full_rows(),
    full_columns: extent.full_columns(),
    local_row_start: extent.row_start(),
    local_rows: extent.row_count(),
    local_column_start: extent.column_start(),
    local_columns: extent.column_count(),
    region_offset_bytes: region.offset_bytes(),
    region_byte_length: region.byte_length(),
  }
}

///|
fn validate_region_order(
  rank_plan : @tensor_parallel_plan.TensorParallelRankPlan,
) -> Unit raise TensorParallelDevicePlanError {
  let mut previous_end = 0L
  let mut materialized = 0L
  for reference in 0..<rank_plan.tensor_count() {
    let tensor = tensor_reference(reference)
    guard rank_plan.tensor(tensor) is Some(placement) else {
      raise invalid(TensorReference, reference)
    }
    let region = placement.region()
    if region.offset_bytes() < previous_end ||
      materialized > MAX_INT64 - region.byte_length() {
      raise invalid(TensorRegion, reference)
    }
    materialized += region.byte_length()
    previous_end = region.end_offset_bytes()
  }
  if materialized != rank_plan.materialized_bytes() ||
    previous_end != rank_plan.arena_bytes() {
    raise invalid(TensorRegion, rank_plan.tensor_count())
  }
}

///|
fn validate_collectives(
  rank_plan : @tensor_parallel_plan.TensorParallelRankPlan,
  operations : ReadOnlyArray[@tensor_parallel_plan.TensorParallelOperationPlan],
) -> ReadOnlyArray[@tensor_parallel_plan.TensorCollectiveSite] raise TensorParallelDevicePlanError {
  let topology = rank_plan.topology()
  if topology.world_size() == 1 && rank_plan.collective_count() != 0 {
    raise invalid(CollectiveCount, rank_plan.collective_count())
  }
  let output : Array[@tensor_parallel_plan.TensorCollectiveSite] = []
  let mut previous_operation = -1
  for index in 0..<rank_plan.collective_count() {
    guard rank_plan.collective(index) is Some(site) else {
      raise invalid(CollectiveCount, index)
    }
    let operation_index = site.operation().as_int()
    if site.sequence().as_int() != index + 1 {
      raise invalid(CollectiveSequence, index)
    }
    if operation_index <= previous_operation ||
      operation_index < 0 ||
      operation_index >= operations.length() {
      raise invalid(CollectiveOperation, index)
    }
    let geometry = operations[operation_index]
    if geometry.operation().as_int() != operation_index {
      raise invalid(CollectiveOperation, index)
    }
    match site.kind() {
      SumAllReduce =>
        if site.full_width() != geometry.output_width() ||
          site.local_width() != geometry.output_width() {
          raise invalid(CollectiveWidth, index)
        }
      LastAxisAllGather =>
        if topology.world_size() <= 1 ||
          site.local_width() > 2147483647 / topology.world_size() ||
          site.full_width() != site.local_width() * topology.world_size() ||
          site.local_width() != geometry.output_width() {
          raise invalid(CollectiveWidth, index)
        }
    }
    output.push(site)
    previous_operation = operation_index
  }
  ReadOnlyArray::from_array(output)
}
