///|
const MAX_INT64 : Int64 = 9223372036854775807L

///|
const MAX_OPERATIONS : Int = 4096

///|
const MAX_ACTIVATION_INPUTS : Int = 8192

///|
const BF16_BYTES : Int64 = 2L

///|
priv struct FreeBlock {
  offset : Int64
  byte_length : Int64
}

///|
priv struct ActiveBlock {
  offset : Int64
  byte_length : Int64
  live_through_step : Int
}

///|
fn checked_add(
  left : Int64,
  right : Int64,
  operation : @plan.OperationId,
) -> Int64 raise DeviceMemoryError {
  if left < 0L || right < 0L || left > MAX_INT64 - right {
    raise SizeOverflow(operation)
  }
  left + right
}

///|
fn checked_multiply(
  left : Int64,
  right : Int64,
  operation : @plan.OperationId,
) -> Int64 raise DeviceMemoryError {
  if left < 0L || right < 0L || (right != 0L && left > MAX_INT64 / right) {
    raise SizeOverflow(operation)
  }
  left * right
}

///|
fn checked_align(
  value : Int64,
  alignment : Int64,
  operation : @plan.OperationId,
) -> Int64 raise DeviceMemoryError {
  let remainder = value % alignment
  if remainder == 0L {
    value
  } else {
    checked_add(value, alignment - remainder, operation)
  }
}

///|
fn output_width(
  operation : @device_plan.StaticOperation,
) -> Int64 raise DeviceMemoryError {
  let width = match (operation.kind(), operation.shape()) {
    (Embedding, EmbeddingShape(width~, ..)) => Int64::from_int(width)
    (RmsNorm, NormalizationShape(width~, ..)) => Int64::from_int(width)
    (QkvProjection, ProjectionShape(output_width~, ..)) =>
      Int64::from_int(output_width)
    (RotaryEmbedding, RotaryShape(query_heads~, kv_heads~, head_dimension~, ..)) => {
      let heads = checked_add(
        Int64::from_int(query_heads),
        checked_multiply(Int64::from_int(kv_heads), 2L, operation.id()),
        operation.id(),
      )
      checked_multiply(heads, Int64::from_int(head_dimension), operation.id())
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
    _ => raise InvalidOperation(operation.id(), ShapeMismatch)
  }
  if width <= 0L {
    raise InvalidOperation(operation.id(), ShapeMismatch)
  }
  width
}

///|
fn activation_sizes(
  operation : @device_plan.StaticOperation,
  token_rows : Int64,
  limits : DeviceMemoryLimits,
) -> (Int64, Int64) raise DeviceMemoryError {
  let elements = checked_multiply(
    output_width(operation),
    token_rows,
    operation.id(),
  )
  let logical = checked_multiply(elements, BF16_BYTES, operation.id())
  let reserved = checked_align(
    logical,
    limits.activation_alignment(),
    operation.id(),
  )
  (logical, reserved)
}

///|
fn insert_free_block(free : Array[FreeBlock], block : FreeBlock) -> Unit {
  let mut position = 0
  while position < free.length() && free[position].offset < block.offset {
    position += 1
  }
  free.insert(position, block)
  if position > 0 {
    let previous = free[position - 1]
    let current = free[position]
    if previous.offset + previous.byte_length == current.offset {
      free[position - 1] = {
        offset: previous.offset,
        byte_length: previous.byte_length + current.byte_length,
      }
      ignore(free.remove(position))
      position -= 1
    }
  }
  if position + 1 < free.length() {
    let current = free[position]
    let next = free[position + 1]
    if current.offset + current.byte_length == next.offset {
      free[position] = {
        offset: current.offset,
        byte_length: current.byte_length + next.byte_length,
      }
      ignore(free.remove(position + 1))
    }
  }
}

///|
fn release_completed(
  active : Array[ActiveBlock],
  free : Array[FreeBlock],
  step : Int,
) -> Array[ActiveBlock] {
  let retained : Array[ActiveBlock] = []
  for block in active {
    if block.live_through_step < step {
      insert_free_block(free, {
        offset: block.offset,
        byte_length: block.byte_length,
      })
    } else {
      retained.push(block)
    }
  }
  retained
}

///|
fn take_best_fit(free : Array[FreeBlock], required : Int64) -> Int64? {
  let mut selected = -1
  for index, block in free {
    if block.byte_length >= required &&
      (
        selected < 0 ||
        block.byte_length < free[selected].byte_length ||
        (
          block.byte_length == free[selected].byte_length &&
          block.offset < free[selected].offset
        )
      ) {
      selected = index
    }
  }
  if selected < 0 {
    None
  } else {
    let block = free.remove(selected)
    if block.byte_length > required {
      insert_free_block(free, {
        offset: block.offset + required,
        byte_length: block.byte_length - required,
      })
    }
    Some(block.offset)
  }
}

///|
fn slot_for_offset(slot_offsets : Array[Int64], offset : Int64) -> Int {
  for index, candidate in slot_offsets {
    if candidate == offset {
      return index
    }
  }
  slot_offsets.push(offset)
  slot_offsets.length() - 1
}

///|
fn region_for(
  regions : Array[ActivationRegion],
  reference : @plan.ActivationRef,
) -> ActivationRegion raise DeviceMemoryError {
  guard regions.get(reference.as_int()) is Some(region) else {
    raise InvalidActivation(reference, ActivationInput)
  }
  region
}

///|
fn workspace_region(
  static_plan : @device_plan.StaticDevicePlan,
  activation_bytes : Int64,
) -> WorkspaceRegion raise DeviceMemoryError {
  let operations = static_plan.operations()
  let operation = operations[operations.length() - 1].id()
  let alignment = static_plan.workspace_alignment()
  let bytes = static_plan.workspace_bytes()
  if alignment <= 0L ||
    (alignment & (alignment - 1L)) != 0L ||
    bytes < 0L ||
    bytes % alignment != 0L {
    raise InvalidPlan(Workspace)
  }
  {
    offset_bytes: checked_align(activation_bytes, alignment, operation),
    byte_length: bytes,
    alignment_bytes: alignment,
  }
}

///|
fn token_row_envelope(
  static_plan : @device_plan.StaticDevicePlan,
) -> Int64 raise DeviceMemoryError {
  let constraints = static_plan.shape_constraints()
  let operation = static_plan.operations()[0].id()
  let batch_rows = Int64::from_int(constraints.max_batch_rows())
  let sequence_tokens = Int64::from_int(constraints.max_sequence_tokens())
  if batch_rows <= 0L || sequence_tokens <= 0L {
    raise InvalidPlan(ShapeEnvelope)
  }
  checked_multiply(batch_rows, sequence_tokens, operation)
}

///|
fn build_operation_bindings(
  static_plan : @device_plan.StaticDevicePlan,
  regions : Array[ActivationRegion],
  workspace : WorkspaceRegion,
) -> ReadOnlyArray[OperationMemoryBinding] raise DeviceMemoryError {
  let bindings : Array[OperationMemoryBinding] = []
  for operation in static_plan.operations() {
    let inputs : Array[ActivationRegion] = []
    for input in operation.inputs() {
      match input {
        ActivationValue(reference) =>
          inputs.push(region_for(regions, reference))
        ExternalInput(_) | WeightRegion(_) => ()
      }
    }
    let outputs : Array[ActivationRegion] = []
    for output in operation.outputs() {
      outputs.push(region_for(regions, output))
    }
    bindings.push({
      operation_id: operation.id(),
      activation_inputs: ReadOnlyArray::from_array(inputs),
      activation_outputs: ReadOnlyArray::from_array(outputs),
      workspace,
    })
  }
  ReadOnlyArray::from_array(bindings)
}

///|
/// Plans reusable BF16 activation slots and one shared aligned workspace.
///
/// Input and output lifetimes overlap at their operation boundary, so a slot
/// is released only after its final consuming operation. The final output is
/// retained through the completion step. Planning is deterministic and all
/// arithmetic and the complete arena are checked before device allocation.
pub fn plan(
  static_plan : @device_plan.StaticDevicePlan,
  limits : DeviceMemoryLimits,
) -> DeviceMemoryPlan raise DeviceMemoryError {
  let operations = static_plan.operations()
  if operations.length() == 0 {
    raise InvalidPlan(EmptyOperations)
  }
  if operations.length() > MAX_OPERATIONS {
    raise InvalidPlan(OperationCount)
  }
  if !static_plan.target().supports_bf16() {
    raise InvalidPlan(TargetDtype)
  }
  let token_rows = token_row_envelope(static_plan)
  let logical_sizes : Array[Int64] = []
  let reserved_sizes : Array[Int64] = []
  let last_uses : Array[Int] = []
  let mut activation_input_count = 0
  for index, operation in operations {
    if operation.id().as_int() != index {
      raise InvalidOperation(operation.id(), OperationOrder)
    }
    if operation.outputs().length() != 1 {
      raise InvalidOperation(operation.id(), OutputArity)
    }
    for input in operation.inputs() {
      match input {
        ActivationValue(reference) => {
          activation_input_count += 1
          if activation_input_count > MAX_ACTIVATION_INPUTS {
            raise InvalidPlan(ActivationInputCount)
          }
          let source = reference.as_int()
          if source < 0 || source >= last_uses.length() {
            raise InvalidOperation(operation.id(), ActivationInput)
          }
          last_uses[source] = index
        }
        ExternalInput(_) | WeightRegion(_) => ()
      }
    }
    let output = operation.outputs()[0]
    if output.as_int() != last_uses.length() {
      raise InvalidOperation(operation.id(), OutputOrder)
    }
    let (logical, reserved) = activation_sizes(operation, token_rows, limits)
    logical_sizes.push(logical)
    reserved_sizes.push(reserved)
    last_uses.push(index)
  }
  let final_index = static_plan.final_output().as_int()
  if final_index < 0 || final_index >= last_uses.length() {
    raise InvalidActivation(static_plan.final_output(), FinalOutput)
  }
  last_uses[final_index] = operations.length()
  let free : Array[FreeBlock] = []
  let mut active : Array[ActiveBlock] = []
  let slot_offsets : Array[Int64] = []
  let regions : Array[ActivationRegion] = []
  let mut high_water = 0L
  for index, operation in operations {
    active = release_completed(active, free, index)
    let required = reserved_sizes[index]
    let offset = match take_best_fit(free, required) {
      Some(offset) => offset
      None => {
        let offset = checked_align(
          high_water,
          limits.activation_alignment(),
          operation.id(),
        )
        high_water = checked_add(offset, required, operation.id())
        if high_water > limits.max_arena_bytes() {
          raise ArenaLimitExceeded
        }
        offset
      }
    }
    active.push({
      offset,
      byte_length: required,
      live_through_step: last_uses[index],
    })
    let reference = operation.outputs()[0]
    regions.push({
      reference,
      slot: { value: slot_for_offset(slot_offsets, offset) },
      offset_bytes: offset,
      logical_bytes: logical_sizes[index],
      reserved_bytes: required,
      produced_by: operation.id(),
      live_through_step: last_uses[index],
      retained_output: reference == static_plan.final_output(),
    })
  }
  let workspace = workspace_region(static_plan, high_water)
  let last_operation = operations[operations.length() - 1].id()
  let arena_bytes = checked_add(
    workspace.offset_bytes(),
    workspace.byte_length(),
    last_operation,
  )
  if arena_bytes > limits.max_arena_bytes() {
    raise ArenaLimitExceeded
  }
  {
    identity: static_plan.identity(),
    target: static_plan.target(),
    limits,
    token_rows,
    activation_bytes: high_water,
    workspace,
    arena_bytes,
    slot_count: slot_offsets.length(),
    regions: ReadOnlyArray::from_array(regions),
    bindings: build_operation_bindings(static_plan, regions, workspace),
    final_output: static_plan.final_output(),
  }
}
