///|
/// Cross-authenticates the semantic operation sequence against the exact
/// operation-local rank geometry emitted by the model-plan builder. This
/// bridge never reconstructs a model-family graph or derives widths by kind.
fn validate_operation_plans(
  model : @plan.ModelPlan,
  rank_plan : @tensor_parallel_plan.TensorParallelRankPlan,
) -> ReadOnlyArray[@tensor_parallel_plan.TensorParallelOperationPlan] raise TensorParallelLaunchContractError {
  let semantic = model.operations()
  if semantic.is_empty() || rank_plan.operation_count() != semantic.length() {
    raise invalid(OperationCount, operation=rank_plan.operation_count())
  }
  let output : Array[@tensor_parallel_plan.TensorParallelOperationPlan] = []
  for index, operation in semantic {
    guard rank_plan.operation(operation.id()) is Some(geometry) else {
      raise invalid(OperationOrder, operation=index)
    }
    if geometry.operation().as_int() != index ||
      geometry.kind() != operation.kind() ||
      geometry.capability() != operation.required_capability() ||
      geometry.input_width() <= 0 ||
      geometry.output_width() <= 0 ||
      (geometry.kind() == GatedMlp && geometry.inner_width() <= 0) ||
      (geometry.kind() != GatedMlp && geometry.inner_width() != 0) {
      raise invalid(OperationShape, operation=index)
    }
    let no_heads = geometry.query_head_count() == 0 &&
      geometry.kv_head_count() == 0 &&
      geometry.head_dimension() == 0
    let valid_heads = geometry.query_head_count() > 0 &&
      geometry.kv_head_count() > 0 &&
      geometry.head_dimension() > 0 &&
      geometry.query_head_count() >= geometry.kv_head_count() &&
      geometry.query_head_count() % geometry.kv_head_count() == 0
    if !no_heads && !valid_heads {
      raise invalid(OperationShape, operation=index)
    }
    output.push(geometry)
  }
  ReadOnlyArray::from_array(output)
}
