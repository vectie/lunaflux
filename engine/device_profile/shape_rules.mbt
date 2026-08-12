///|
fn operation_output_width(
  operation : @device_plan.StaticOperation,
) -> Int64 raise DeviceProfileError {
  let width = match (operation.kind(), operation.shape()) {
    (Embedding, EmbeddingShape(width~, ..)) => Int64::from_int(width)
    (RmsNorm, NormalizationShape(width~, ..)) => Int64::from_int(width)
    (QkvProjection, ProjectionShape(output_width~, ..)) =>
      Int64::from_int(output_width)
    (RotaryEmbedding, RotaryShape(query_heads~, kv_heads~, head_dimension~, ..)) => {
      let operation_id = operation.id()
      let kv_twice = checked_multiply(
        Int64::from_int(kv_heads),
        2L,
        operation_id,
      )
      checked_multiply(
        checked_add(Int64::from_int(query_heads), kv_twice, operation_id),
        Int64::from_int(head_dimension),
        operation_id,
      )
    }
    (CausalAttention, AttentionShape(query_heads~, head_dimension~, ..)) =>
      checked_multiply(
        Int64::from_int(query_heads),
        Int64::from_int(head_dimension),
        operation.id(),
      )
    (OutputProjection | LanguageModelHead, ProjectionShape(output_width~, ..)) =>
      Int64::from_int(output_width)
    (ResidualAdd, ElementwiseShape(width~)) => Int64::from_int(width)
    (GatedMlp, GatedMlpShape(hidden_size~, ..)) => Int64::from_int(hidden_size)
    _ => raise InvalidOperation(operation.id(), Shape)
  }
  if width <= 0L {
    raise InvalidOperation(operation.id(), Shape)
  }
  width
}

///|
fn expected_activation_widths(
  operation : @device_plan.StaticOperation,
) -> ReadOnlyArray[Int64] raise DeviceProfileError {
  let widths : Array[Int64] = match (operation.kind(), operation.shape()) {
    (Embedding, EmbeddingShape(..)) => []
    (RmsNorm, NormalizationShape(width~, ..)) => [Int64::from_int(width)]
    (QkvProjection, ProjectionShape(input_width~, ..)) =>
      [Int64::from_int(input_width)]
    (RotaryEmbedding, RotaryShape(query_heads~, kv_heads~, head_dimension~, ..)) => {
      let operation_id = operation.id()
      let total_heads = checked_add(
        Int64::from_int(query_heads),
        checked_multiply(Int64::from_int(kv_heads), 2L, operation_id),
        operation_id,
      )
      [
        checked_multiply(
          total_heads,
          Int64::from_int(head_dimension),
          operation_id,
        ),
      ]
    }
    (CausalAttention, AttentionShape(query_heads~, kv_heads~, head_dimension~)) => {
      let operation_id = operation.id()
      let total_heads = checked_add(
        Int64::from_int(query_heads),
        checked_multiply(Int64::from_int(kv_heads), 2L, operation_id),
        operation_id,
      )
      [
        checked_multiply(
          total_heads,
          Int64::from_int(head_dimension),
          operation_id,
        ),
      ]
    }
    (OutputProjection | LanguageModelHead, ProjectionShape(input_width~, ..)) =>
      [Int64::from_int(input_width)]
    (ResidualAdd, ElementwiseShape(width~)) => {
      let width = Int64::from_int(width)
      [width, width]
    }
    (GatedMlp, GatedMlpShape(hidden_size~, ..)) =>
      [Int64::from_int(hidden_size)]
    _ => raise InvalidOperation(operation.id(), Shape)
  }
  for width in widths {
    if width <= 0L {
      raise InvalidOperation(operation.id(), Shape)
    }
  }
  ReadOnlyArray::from_array(widths)
}

///|
fn valid_input_roles(operation : @device_plan.StaticOperation) -> Bool {
  let inputs = operation.inputs()
  match operation.kind() {
    Embedding =>
      inputs.length() == 2 &&
      inputs[0] is ExternalInput(TokenIds) &&
      inputs[1] is WeightRegion(_)
    RmsNorm | OutputProjection | LanguageModelHead =>
      inputs.length() == 2 &&
      inputs[0] is ActivationValue(_) &&
      inputs[1] is WeightRegion(_)
    QkvProjection | GatedMlp =>
      inputs.length() == 4 &&
      inputs[0] is ActivationValue(_) &&
      inputs[1] is WeightRegion(_) &&
      inputs[2] is WeightRegion(_) &&
      inputs[3] is WeightRegion(_)
    RotaryEmbedding | CausalAttention =>
      inputs.length() == 1 && inputs[0] is ActivationValue(_)
    ResidualAdd =>
      inputs.length() == 2 &&
      inputs[0] is ActivationValue(_) &&
      inputs[1] is ActivationValue(_)
  }
}
