///|
fn complementary(
  left : @tensor_parallel_plan.TensorRankPlacement,
  right : @tensor_parallel_plan.TensorRankPlacement,
) -> Bool {
  let a = left.extent()
  let b = right.extent()
  if left.placement() != right.placement() ||
    a.full_rows() != b.full_rows() ||
    a.full_columns() != b.full_columns() {
    return false
  }
  match left.placement() {
    Replicated =>
      a == b &&
      a.row_start() == 0L &&
      a.column_start() == 0L &&
      a.row_count() == a.full_rows() &&
      a.column_count() == a.full_columns()
    ShardRows =>
      a.row_start() == 0L &&
      b.row_start() == a.row_count() &&
      b.row_count() == a.full_rows() - a.row_count() &&
      a.row_count() == b.row_count() &&
      a.column_start() == 0L &&
      b.column_start() == 0L &&
      a.column_count() == a.full_columns() &&
      b.column_count() == b.full_columns()
    ShardColumns =>
      a.column_start() == 0L &&
      b.column_start() == a.column_count() &&
      b.column_count() == a.full_columns() - a.column_count() &&
      a.column_count() == b.column_count() &&
      a.row_start() == 0L &&
      b.row_start() == 0L &&
      a.row_count() == a.full_rows() &&
      b.row_count() == b.full_rows()
  }
}

///|
fn validate_pair(
  model : @plan.ModelPlan,
  left : @tensor_parallel_plan.TensorParallelRankPlan,
  right : @tensor_parallel_plan.TensorParallelRankPlan,
) -> Unit raise TwoNodePlanError {
  if left.tensor_count() != right.tensor_count() ||
    left.tensor_count() != model.tensor_count() {
    raise Rejected(rule=TensorPartition, rank=-1)
  }
  for index in 0..<left.tensor_count() {
    let reference = @plan.TensorRef::new(index) catch {
      _ => raise Rejected(rule=TensorPartition, rank=-1)
    }
    guard left.tensor(reference) is Some(a) &&
      right.tensor(reference) is Some(b) &&
      model.tensor(reference) is Some(tensor) else {
      raise Rejected(rule=TensorPartition, rank=-1)
    }
    let dimensions = tensor.shape().dimensions()
    let full_shape = match dimensions[:] {
      [columns] => (1L, columns)
      [rows, columns] => (rows, columns)
      _ => raise Rejected(rule=TensorPartition, rank=-1)
    }
    if a.extent().full_rows() != full_shape.0 ||
      a.extent().full_columns() != full_shape.1 {
      raise Rejected(rule=TensorPartition, rank=-1)
    }
    let storage = tensor.storage()
    if storage.dtype() != @numeric_contract.StorageDType::bfloat16() ||
      storage.encoding() != @numeric_contract.StorageEncoding::plain_v1() ||
      storage.layout() != @numeric_contract.StorageLayout::dense_row_major_v1() {
      raise Rejected(rule=NumericFormat, rank=-1)
    }
    if !complementary(a, b) {
      raise Rejected(rule=TensorPartition, rank=-1)
    }
    // This workstream initially carries the existing BF16 rank ABI only.
    let elements = checked_multiply(
      a.extent().row_count(),
      a.extent().column_count(),
    )
    if a.extent().byte_length() != checked_multiply(elements, 2L) ||
      b.extent().byte_length() != a.extent().byte_length() {
      raise Rejected(rule=NumericFormat, rank=-1)
    }
    validate_transfer_pair(a, b)
  }
  if left.operation_count() != right.operation_count() ||
    left.operation_count() != model.operations().length() {
    raise Rejected(rule=OperationMismatch, rank=-1)
  }
  for operation in model.operations() {
    guard left.operation(operation.id()) is Some(a) &&
      right.operation(operation.id()) is Some(b) else {
      raise Rejected(rule=OperationMismatch, rank=-1)
    }
    if a != b {
      raise Rejected(rule=OperationMismatch, rank=-1)
    }
  }
  if left.collective_count() != right.collective_count() ||
    left.collective_count() == 0 {
    raise Rejected(rule=CollectiveMismatch, rank=-1)
  }
  for index in 0..<left.collective_count() {
    if left.collective(index) != right.collective(index) {
      raise Rejected(rule=CollectiveMismatch, rank=-1)
    }
  }
}
