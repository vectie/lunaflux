///|
fn checked_blueprint_product(
  left : Int64,
  right : Int64,
) -> Int64 raise DeviceStepError {
  if left <= 0L || right <= 0L || left > 9223372036854775807L / right {
    raise blueprint_error(ActivationRegion)
  }
  left * right
}

///|
fn blueprint_output_width(
  operation : @plan.PlanOperation,
) -> Int64 raise DeviceStepError {
  let width = match operation.shape() {
    EmbeddingShape(width~, ..)
    | NormalizationShape(width~, ..)
    | ElementwiseShape(width~) => Int64::from_int(width)
    ProjectionShape(output_width~, ..) => Int64::from_int(output_width)
    RotaryShape(query_heads~, kv_heads~, head_dimension~, ..) =>
      checked_blueprint_product(
        Int64::from_int(query_heads + 2 * kv_heads),
        Int64::from_int(head_dimension),
      )
    QkNormalizationShape(query_heads~, kv_heads~, head_dimension~, ..) =>
      checked_blueprint_product(
        Int64::from_int(query_heads + 2 * kv_heads),
        Int64::from_int(head_dimension),
      )
    AttentionShape(query_heads~, head_dimension~, ..) =>
      checked_blueprint_product(
        Int64::from_int(query_heads),
        Int64::from_int(head_dimension),
      )
    GatedMlpShape(hidden_size~, ..) => Int64::from_int(hidden_size)
  }
  if width <= 0L {
    raise blueprint_operation_error(operation.id(), ActivationRegion)
  }
  width
}

///|
fn exact_activation_references(
  semantic : @plan.PlanOperation,
  binding : @device_memory.OperationMemoryBinding,
) -> Bool {
  let expected_inputs : Array[@plan.ActivationRef] = []
  for input in semantic.inputs() {
    if input is Activation(reference) {
      expected_inputs.push(reference)
    }
  }
  let actual_inputs = binding.activation_inputs()
  if expected_inputs.length() != actual_inputs.length() {
    return false
  }
  for index, reference in expected_inputs {
    if actual_inputs[index].reference() != reference {
      return false
    }
  }
  let expected_outputs = semantic.outputs()
  let actual_outputs = binding.activation_outputs()
  if expected_outputs.length() != actual_outputs.length() {
    return false
  }
  for index, reference in expected_outputs {
    if actual_outputs[index].reference() != reference {
      return false
    }
  }
  true
}

///|
fn expected_activation_last_uses(
  model_plan : @plan.ModelPlan,
) -> Array[Int] raise DeviceStepError {
  let operations = model_plan.operations()
  let last_uses = Array::make(operations.length(), -1)
  for index, operation in operations {
    for input in operation.inputs() {
      if input is Activation(reference) {
        let source = reference.as_int()
        if source < 0 || source >= index {
          raise blueprint_operation_error(operation.id(), ActivationRegion)
        }
        last_uses[source] = index
      }
    }
    guard operation.outputs() is [output] && output.as_int() == index else {
      raise blueprint_operation_error(operation.id(), ActivationRegion)
    }
    last_uses[index] = index
  }
  let final_index = model_plan.final_output().as_int()
  if final_index < 0 || final_index >= last_uses.length() {
    raise blueprint_error(ActivationRegion)
  }
  last_uses[final_index] = operations.length()
  last_uses
}

///|
fn byte_ranges_overlap(
  left : @device_memory.ActivationRegion,
  right : @device_memory.ActivationRegion,
) -> Bool {
  left.offset_bytes() < right.end_offset_bytes() &&
  right.offset_bytes() < left.end_offset_bytes()
}

///|
fn live_intervals_overlap(
  left : @device_memory.ActivationRegion,
  right : @device_memory.ActivationRegion,
) -> Bool {
  let left_start = left.produced_by().as_int()
  let right_start = right.produced_by().as_int()
  left_start <= right.live_through_step() &&
  right_start <= left.live_through_step()
}

///|
fn validate_blueprint_memory_plan(
  model_plan : @plan.ModelPlan,
  memory_plan : @device_memory.DeviceMemoryPlan,
) -> Unit raise DeviceStepError {
  let operations = model_plan.operations()
  let shapes = model_plan.shape_constraints()
  let token_rows = checked_blueprint_product(
    shapes.max_batch_rows().to_int64(),
    shapes.max_sequence_tokens().to_int64(),
  )
  let workspace = memory_plan.workspace()
  let workspace_alignment = workspace.alignment_bytes()
  if memory_plan.token_rows() != token_rows ||
    memory_plan.operation_count() != operations.length() ||
    memory_plan.activation_count() != operations.length() ||
    memory_plan.final_output() != model_plan.final_output() ||
    memory_plan.activation_bytes() <= 0L ||
    memory_plan.activation_bytes() > workspace.offset_bytes() ||
    workspace.byte_length() < 0L ||
    workspace_alignment <= 0L ||
    workspace_alignment > 65536L ||
    (workspace_alignment & (workspace_alignment - 1L)) != 0L ||
    workspace.offset_bytes() % workspace_alignment != 0L ||
    workspace.end_offset_bytes() != memory_plan.arena_bytes() ||
    memory_plan.arena_bytes() > memory_plan.limits().max_arena_bytes() {
    raise blueprint_error(ActivationRegion)
  }
  let last_uses = expected_activation_last_uses(model_plan)
  let regions : Array[@device_memory.ActivationRegion] = []
  for index, operation in operations {
    guard operation.outputs() is [reference] else {
      raise blueprint_operation_error(operation.id(), ActivationRegion)
    }
    guard memory_plan.operation_binding(operation.id()) is Some(binding) &&
      binding.operation_id() == operation.id() &&
      binding.workspace() == workspace &&
      exact_activation_references(operation, binding) &&
      memory_plan.region(reference) is Some(region) else {
      raise blueprint_operation_error(operation.id(), ActivationRegion)
    }
    let expected_bytes = checked_blueprint_product(
      checked_blueprint_product(token_rows, blueprint_output_width(operation)),
      2L,
    )
    let retained = region.reference() == model_plan.final_output()
    if region.reference().as_int() != index ||
      region.produced_by() != operation.id() ||
      region.logical_bytes() != expected_bytes ||
      region.reserved_bytes() < expected_bytes ||
      region.offset_bytes() < 0L ||
      region.offset_bytes() % memory_plan.limits().activation_alignment() != 0L ||
      region.end_offset_bytes() > memory_plan.activation_bytes() ||
      region.live_through_step() != last_uses[index] ||
      region.is_retained_output() != retained {
      raise blueprint_operation_error(operation.id(), ActivationRegion)
    }
    regions.push(region)
  }
  for left in 0..<regions.length() {
    for right in (left + 1)..<regions.length() {
      if byte_ranges_overlap(regions[left], regions[right]) &&
        live_intervals_overlap(regions[left], regions[right]) {
        raise blueprint_error(ActivationRegion)
      }
    }
  }
}
