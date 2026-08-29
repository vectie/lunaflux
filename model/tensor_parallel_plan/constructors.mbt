///|
pub fn TensorParallelTopology::new(
  rank~ : Int,
  world_size~ : Int,
) -> TensorParallelTopology raise TensorParallelPlanError {
  if world_size <= 0 || world_size > 16 || rank < 0 || rank >= world_size {
    raise invalid(Topology)
  }
  { rank, world_size }
}

///|
pub fn TensorShardExtent::new(
  full_rows~ : Int64,
  full_columns~ : Int64,
  row_start~ : Int64,
  row_count~ : Int64,
  column_start~ : Int64,
  column_count~ : Int64,
  byte_length~ : Int64,
) -> TensorShardExtent raise TensorParallelPlanError {
  if full_rows <= 0L ||
    full_columns <= 0L ||
    row_start < 0L ||
    row_count <= 0L ||
    column_start < 0L ||
    column_count <= 0L ||
    byte_length <= 0L ||
    checked_add(row_start, row_count) > full_rows ||
    checked_add(column_start, column_count) > full_columns {
    raise invalid(Extent)
  }
  {
    full_rows,
    full_columns,
    row_start,
    row_count,
    column_start,
    column_count,
    byte_length,
  }
}

///|
pub fn TensorArenaRegion::new(
  offset_bytes~ : Int64,
  byte_length~ : Int64,
  alignment_bytes~ : Int64,
  max_arena_bytes~ : Int64,
) -> TensorArenaRegion raise TensorParallelPlanError {
  if offset_bytes < 0L ||
    byte_length <= 0L ||
    !valid_alignment(alignment_bytes) ||
    offset_bytes % alignment_bytes != 0L ||
    max_arena_bytes <= 0L {
    raise invalid(Region)
  }
  let end = checked_add(offset_bytes, byte_length)
  if end > max_arena_bytes {
    raise invalid(ArenaBytes)
  }
  { offset_bytes, byte_length, alignment_bytes }
}

///|
pub fn TensorTransferRecipe::new(
  reference~ : @plan.TensorRef,
  source_base_bytes~ : Int64,
  source_stride_bytes~ : Int64,
  destination_base_bytes~ : Int64,
  destination_stride_bytes~ : Int64,
  segment_bytes~ : Int64,
  segment_count~ : Int64,
  destination_region~ : TensorArenaRegion,
) -> TensorTransferRecipe raise TensorParallelPlanError {
  let index = reference.as_int()
  if source_base_bytes < 0L ||
    source_stride_bytes < 0L ||
    destination_base_bytes != destination_region.offset_bytes ||
    destination_stride_bytes < 0L ||
    segment_bytes <= 0L ||
    segment_count <= 0L {
    raise invalid(TransferRecipe, index~)
  }
  if segment_count > 1L &&
    (
      source_stride_bytes < segment_bytes ||
      destination_stride_bytes < segment_bytes
    ) {
    raise invalid(TransferRecipe, index~)
  }
  let last = segment_count - 1L
  let source_end = checked_add(
    checked_add(
      source_base_bytes,
      checked_multiply(last, source_stride_bytes, index~),
      index~,
    ),
    segment_bytes,
    index~,
  )
  ignore(source_end)
  let destination_end = checked_add(
    checked_add(
      destination_base_bytes,
      checked_multiply(last, destination_stride_bytes, index~),
      index~,
    ),
    segment_bytes,
    index~,
  )
  if destination_end > destination_region.end_offset_bytes() ||
    checked_multiply(segment_bytes, segment_count, index~) !=
    destination_region.byte_length() {
    raise invalid(TransferRecipe, index~)
  }
  {
    reference,
    source_base_bytes,
    source_stride_bytes,
    destination_base_bytes,
    destination_stride_bytes,
    segment_bytes,
    segment_count,
  }
}

///|
pub fn TensorRankPlacement::new(
  reference~ : @plan.TensorRef,
  placement~ : TensorPlacement,
  extent~ : TensorShardExtent,
  region~ : TensorArenaRegion,
  transfer_recipe~ : TensorTransferRecipe,
) -> TensorRankPlacement raise TensorParallelPlanError {
  let index = reference.as_int()
  if transfer_recipe.reference != reference ||
    extent.byte_length != region.byte_length {
    raise invalid(Extent, index~)
  }
  match placement {
    Replicated =>
      if extent.row_start != 0L ||
        extent.row_count != extent.full_rows ||
        extent.column_start != 0L ||
        extent.column_count != extent.full_columns {
        raise invalid(Extent, index~)
      }
    ShardRows =>
      if extent.column_start != 0L ||
        extent.column_count != extent.full_columns ||
        extent.row_count >= extent.full_rows {
        raise invalid(Extent, index~)
      }
    ShardColumns =>
      if extent.row_start != 0L ||
        extent.row_count != extent.full_rows ||
        extent.column_count >= extent.full_columns {
        raise invalid(Extent, index~)
      }
  }
  { reference, placement, extent, region, recipe: transfer_recipe }
}

///|
pub fn TensorParallelOperationPlan::new(
  operation~ : @plan.OperationId,
  kind~ : @plan.OperationKind,
  capability~ : @plan.KernelCapabilityId,
  input_width~ : Int,
  output_width~ : Int,
  inner_width~ : Int,
  query_head_count~ : Int,
  kv_head_count~ : Int,
  head_dimension~ : Int,
) -> TensorParallelOperationPlan raise TensorParallelPlanError {
  let index = operation.as_int()
  if kind == QkRmsNorm {
    raise invalid(UnsupportedSemanticOperation, index~)
  }
  if input_width <= 0 ||
    output_width <= 0 ||
    (kind == GatedMlp && inner_width <= 0) ||
    (kind != GatedMlp && inner_width != 0) {
    raise invalid(OperationGeometry, index~)
  }
  let no_heads = query_head_count == 0 &&
    kv_head_count == 0 &&
    head_dimension == 0
  let valid_heads = query_head_count > 0 &&
    kv_head_count > 0 &&
    head_dimension > 0 &&
    query_head_count >= kv_head_count &&
    query_head_count % kv_head_count == 0
  let head_operation = kind
    is (QkvProjection | RotaryEmbedding | CausalAttention)
  if (head_operation && !valid_heads) || (!head_operation && !no_heads) {
    raise invalid(OperationGeometry, index~)
  }
  if head_operation {
    if kv_head_count > (2147483647 - query_head_count) / 2 ||
      query_head_count > 2147483647 / head_dimension {
      raise invalid(OperationGeometry, index~)
    }
    let head_width = query_head_count + 2 * kv_head_count
    if head_width > 2147483647 / head_dimension {
      raise invalid(OperationGeometry, index~)
    }
    let qkv_width = head_width * head_dimension
    let attention_width = query_head_count * head_dimension
    match kind {
      QkvProjection =>
        if output_width != qkv_width {
          raise invalid(OperationGeometry, index~)
        }
      RotaryEmbedding =>
        if input_width != qkv_width || output_width != qkv_width {
          raise invalid(OperationGeometry, index~)
        }
      CausalAttention =>
        if input_width != qkv_width || output_width != attention_width {
          raise invalid(OperationGeometry, index~)
        }
      _ => ()
    }
  }
  {
    operation,
    kind,
    capability,
    input_width,
    output_width,
    inner_width,
    query_head_count,
    kv_head_count,
    head_dimension,
  }
}

///|
pub fn TensorCollectiveSite::new(
  sequence~ : Int,
  operation~ : @plan.OperationId,
  kind~ : TensorCollectiveKind,
  full_width~ : Int,
  local_width~ : Int,
  world_size~ : Int,
) -> TensorCollectiveSite raise TensorParallelPlanError {
  if sequence <= 0 || world_size <= 1 || full_width <= 0 || local_width <= 0 {
    raise invalid(CollectiveGeometry, index=sequence)
  }
  match kind {
    SumAllReduce =>
      if full_width != local_width {
        raise invalid(CollectiveGeometry, index=sequence)
      }
    LastAxisAllGather =>
      if local_width > 2147483647 / world_size ||
        full_width != local_width * world_size {
        raise invalid(CollectiveGeometry, index=sequence)
      }
  }
  {
    sequence: TensorCollectiveSequenceId(sequence),
    operation,
    kind,
    full_width,
    local_width,
  }
}

///|
fn validate_tensor_topology(
  tensor : TensorRankPlacement,
  topology : TensorParallelTopology,
  index : Int,
) -> Unit raise TensorParallelPlanError {
  let extent = tensor.extent
  let world = topology.world_size.to_int64()
  let rank = topology.rank.to_int64()
  match tensor.placement {
    Replicated => ()
    ShardRows => {
      if topology.world_size <= 1 || extent.full_rows % world != 0L {
        raise invalid(Extent, index~)
      }
      let local_rows = extent.full_rows / world
      if extent.row_count != local_rows ||
        extent.row_start != rank * local_rows ||
        extent.column_start != 0L ||
        extent.column_count != extent.full_columns {
        raise invalid(Extent, index~)
      }
    }
    ShardColumns => {
      if topology.world_size <= 1 || extent.full_columns % world != 0L {
        raise invalid(Extent, index~)
      }
      let local_columns = extent.full_columns / world
      if extent.column_count != local_columns ||
        extent.column_start != rank * local_columns ||
        extent.row_start != 0L ||
        extent.row_count != extent.full_rows {
        raise invalid(Extent, index~)
      }
    }
  }
}

///|
fn validate_operation_topology(
  operation : TensorParallelOperationPlan,
  model : @plan.ModelPlan,
  semantic : @plan.PlanOperation,
  topology : TensorParallelTopology,
  index : Int,
) -> Unit raise TensorParallelPlanError {
  let world = topology.world_size
  let mut valid = false
  match semantic.shape() {
    EmbeddingShape(width~, ..) =>
      valid = operation.input_width == 1 &&
        operation.output_width == width &&
        operation.inner_width == 0
    NormalizationShape(width~, ..) | ElementwiseShape(width~) =>
      valid = operation.input_width == width &&
        operation.output_width == width &&
        operation.inner_width == 0
    QkNormalizationShape(..) =>
      raise invalid(UnsupportedSemanticOperation, index~)
    ProjectionShape(input_width~, output_width~) =>
      match semantic.kind() {
        QkvProjection =>
          valid = output_width % world == 0 &&
            operation.input_width == input_width &&
            operation.output_width == output_width / world &&
            operation.inner_width == 0
        OutputProjection =>
          valid = input_width % world == 0 &&
            operation.input_width == input_width / world &&
            operation.output_width == output_width &&
            operation.inner_width == 0
        LanguageModelHead =>
          valid = output_width % world == 0 &&
            operation.input_width == input_width &&
            operation.output_width == output_width / world &&
            operation.inner_width == 0
        _ => valid = false
      }
    RotaryShape(query_heads~, kv_heads~, head_dimension~, ..) =>
      if query_heads % world == 0 && kv_heads % world == 0 {
        valid = operation.query_head_count == query_heads / world &&
          operation.kv_head_count == kv_heads / world &&
          operation.head_dimension == head_dimension
      }
    AttentionShape(query_heads~, kv_heads~, head_dimension~) =>
      if query_heads % world == 0 && kv_heads % world == 0 {
        valid = operation.query_head_count == query_heads / world &&
          operation.kv_head_count == kv_heads / world &&
          operation.head_dimension == head_dimension
      }
    GatedMlpShape(hidden_size~, intermediate_size~) =>
      valid = intermediate_size % world == 0 &&
        operation.input_width == hidden_size &&
        operation.output_width == hidden_size &&
        operation.inner_width == intermediate_size / world
  }
  if semantic.kind() == QkvProjection {
    let shape = model.shape_constraints()
    if shape.attention_head_count() % world != 0 ||
      shape.kv_head_count() % world != 0 ||
      operation.query_head_count != shape.attention_head_count() / world ||
      operation.kv_head_count != shape.kv_head_count() / world ||
      operation.head_dimension != shape.head_dimension() {
      valid = false
    }
  }
  if !valid {
    raise invalid(OperationGeometry, index~)
  }
}

///|
pub fn TensorParallelRankPlan::new(
  model~ : @plan.ModelPlan,
  topology~ : TensorParallelTopology,
  alignment_bytes~ : Int64,
  arena_bytes~ : Int64,
  materialized_bytes~ : Int64,
  tensors~ : ArrayView[TensorRankPlacement],
  operations~ : ArrayView[TensorParallelOperationPlan],
  collectives~ : ArrayView[TensorCollectiveSite],
) -> TensorParallelRankPlan raise TensorParallelPlanError {
  let semantic = model.operations()
  for index, operation in semantic {
    if operation.kind() == QkRmsNorm {
      raise invalid(UnsupportedSemanticOperation, index~)
    }
  }
  if !valid_alignment(alignment_bytes) ||
    arena_bytes <= 0L ||
    materialized_bytes <= 0L ||
    materialized_bytes > arena_bytes ||
    tensors.is_empty() {
    raise invalid(ArenaBytes)
  }
  let mut previous_end = 0L
  let mut materialized = 0L
  for index, tensor in tensors {
    if tensor.reference.as_int() != index ||
      tensor.region.alignment_bytes != alignment_bytes ||
      tensor.region.offset_bytes < previous_end {
      raise invalid(TensorOrder, index~)
    }
    validate_tensor_topology(tensor, topology, index)
    materialized = checked_add(materialized, tensor.region.byte_length, index~)
    previous_end = tensor.region.end_offset_bytes()
  }
  if previous_end != arena_bytes || materialized != materialized_bytes {
    raise invalid(MaterializedBytes)
  }
  if topology.world_size == 1 && !collectives.is_empty() {
    raise invalid(CollectiveOrder)
  }
  if operations.length() != semantic.length() {
    raise invalid(OperationOrder, index=operations.length())
  }
  for index, operation in operations {
    let expected = semantic[index]
    if operation.operation.as_int() != index ||
      expected.id().as_int() != index ||
      operation.kind != expected.kind() ||
      operation.capability != expected.required_capability() {
      raise invalid(OperationOrder, index~)
    }
    validate_operation_topology(operation, model, expected, topology, index)
  }
  let mut previous_operation = -1
  for index, collective in collectives {
    if collective.sequence.0 != index + 1 ||
      collective.operation.as_int() <= previous_operation {
      raise invalid(CollectiveOrder, index~)
    }
    if collective.operation.as_int() >= operations.length() {
      raise invalid(CollectiveOrder, index~)
    }
    let operation = operations[collective.operation.as_int()]
    if collective.kind == SumAllReduce &&
      (
        collective.full_width != operation.output_width ||
        collective.local_width != operation.output_width
      ) {
      raise invalid(CollectiveGeometry, index~)
    }
    if collective.kind == LastAxisAllGather &&
      (
        collective.local_width > 2147483647 / topology.world_size ||
        collective.full_width != collective.local_width * topology.world_size ||
        collective.local_width != operation.output_width
      ) {
      raise invalid(CollectiveGeometry, index~)
    }
    previous_operation = collective.operation.as_int()
  }
  {
    identity: model.identity(),
    topology,
    alignment_bytes,
    arena_bytes,
    materialized_bytes,
    tensors: ReadOnlyArray::from_array(tensors),
    operations: ReadOnlyArray::from_array(operations),
    collectives: ReadOnlyArray::from_array(collectives),
  }
}
