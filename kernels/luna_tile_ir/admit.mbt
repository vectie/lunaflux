///|
fn luna_tile_constraint_id(constraint : LunaTileConstraint) -> Int {
  match constraint {
    LunaTileDimensionEquals(id, _, _, _)
    | LunaTileDimensionMultipleOf(id, _, _, _)
    | LunaTileAlignmentAtLeast(id, _, _) => id.as_int()
  }
}

///|
fn canonical_luna_tile_constraints(
  constraints : ReadOnlyArray[LunaTileConstraint],
) -> ReadOnlyArray[LunaTileConstraint] raise LunaTileError {
  let result : Array[LunaTileConstraint] = []
  for constraint in constraints {
    let key = luna_tile_constraint_id(constraint)
    let mut insertion = result.length()
    while insertion > 0 && luna_tile_constraint_id(result[insertion - 1]) > key {
      insertion -= 1
    }
    if insertion > 0 && luna_tile_constraint_id(result[insertion - 1]) == key {
      raise InvalidProgram(Constraint)
    }
    result.insert(insertion, constraint)
  }
  ReadOnlyArray::from_array(result)
}

///|
fn validate_luna_tile_constraint(
  constraint : LunaTileConstraint,
  tensors : ReadOnlyArray[LunaTileTensorView],
) -> Unit raise LunaTileError {
  match constraint {
    LunaTileDimensionEquals(_, tensor_id, dimension, expected) => {
      let tensor = require_luna_tile_tensor(tensors, tensor_id)
      if dimension < 0 ||
        dimension >= tensor.shape.length() ||
        expected <= 0L ||
        tensor.shape[dimension] != expected {
        raise InvalidProgram(Constraint)
      }
    }
    LunaTileDimensionMultipleOf(_, tensor_id, dimension, divisor) => {
      let tensor = require_luna_tile_tensor(tensors, tensor_id)
      if dimension < 0 ||
        dimension >= tensor.shape.length() ||
        divisor <= 0L ||
        tensor.shape[dimension] % divisor != 0L {
        raise InvalidProgram(Constraint)
      }
    }
    LunaTileAlignmentAtLeast(_, tensor_id, minimum) => {
      let tensor = require_luna_tile_tensor(tensors, tensor_id)
      if !luna_tile_is_power_of_two(minimum) || tensor.alignment_bytes < minimum {
        raise InvalidProgram(Constraint)
      }
    }
  }
}

///|
fn validate_luna_tile_instructions(
  instructions : ReadOnlyArray[LunaTileInstruction],
  tensors : ReadOnlyArray[LunaTileTensorView],
  loop_extents : ReadOnlyArray[Int64],
) -> Unit raise LunaTileError {
  let mut current_stage = 0
  let mut async_pending = false
  for index, instruction in instructions {
    match instruction {
      LunaTilePipelineStage(stage) => {
        if async_pending || stage.as_int() != current_stage + 1 {
          raise InvalidInstruction(index~, rule=PipelineStage)
        }
        current_stage = stage.as_int()
      }
      LunaTileAsyncCopy(source, destination, stage) => {
        if current_stage == 0 || stage.as_int() != current_stage {
          raise InvalidInstruction(index~, rule=PipelineStage)
        }
        validate_luna_tile_copy(
          source, destination, tensors, loop_extents, true,
        )
        async_pending = true
      }
      LunaTileBarrier(stage, _) => {
        if !async_pending || stage.as_int() != current_stage {
          raise InvalidInstruction(index~, rule=PipelineBarrier)
        }
        async_pending = false
      }
      LunaTileCopy(source, destination) => {
        if async_pending {
          raise InvalidInstruction(index~, rule=PipelineBarrier)
        }
        validate_luna_tile_copy(
          source, destination, tensors, loop_extents, false,
        )
      }
      LunaTileMma(destination~, left~, right~, accumulator~) => {
        if async_pending {
          raise InvalidInstruction(index~, rule=PipelineBarrier)
        }
        validate_luna_tile_mma(tensors, destination, left, right, accumulator)
      }
      LunaTileReduce(_, destination~, source~, axis~) => {
        if async_pending {
          raise InvalidInstruction(index~, rule=PipelineBarrier)
        }
        validate_luna_tile_reduce(tensors, destination, source, axis)
      }
      LunaTileElementwise(kind, destination~, inputs~) => {
        if async_pending {
          raise InvalidInstruction(index~, rule=PipelineBarrier)
        }
        validate_luna_tile_elementwise(tensors, kind, destination, inputs)
      }
    }
  }
  if async_pending {
    raise InvalidProgram(PipelineBarrier)
  }
}

///|
fn validate_luna_tile_residual_tensor(
  tensors : ReadOnlyArray[LunaTileTensorView],
  id : LunaTileTensorId,
  width : Int64,
) -> Unit raise LunaTileError {
  let tensor = require_luna_tile_tensor(tensors, id)
  if tensor.dtype != BFloat16 ||
    tensor.address_space != Register ||
    tensor.shape.length() != 1 ||
    tensor.shape[0] != width ||
    tensor.strides_bytes.length() != 1 ||
    tensor.strides_bytes[0] != 2L ||
    tensor.offset_bytes != 0L {
    raise InvalidProgram(SemanticContract)
  }
}

///|
fn validate_luna_tile_residual_add(
  semantic : LunaTileResidualAddSemantics,
  loop_extents : ReadOnlyArray[Int64],
  tensors : ReadOnlyArray[LunaTileTensorView],
  instructions : ReadOnlyArray[LunaTileInstruction],
) -> Unit raise LunaTileError {
  if semantic.width <= 0L ||
    semantic.left == semantic.right ||
    semantic.left == semantic.output ||
    semantic.right == semantic.output ||
    loop_extents.length() != 1 ||
    loop_extents[0] != 1L ||
    tensors.length() != 3 ||
    instructions.length() != 1 {
    raise InvalidProgram(SemanticContract)
  }
  validate_luna_tile_residual_tensor(tensors, semantic.left, semantic.width)
  validate_luna_tile_residual_tensor(tensors, semantic.right, semantic.width)
  validate_luna_tile_residual_tensor(tensors, semantic.output, semantic.width)
  match instructions[0] {
    LunaTileElementwise(Add, destination~, inputs~) =>
      if destination != semantic.output ||
        inputs.length() != 2 ||
        inputs[0] != semantic.left ||
        inputs[1] != semantic.right {
        raise InvalidProgram(SemanticContract)
      }
    _ => raise InvalidProgram(SemanticContract)
  }
}

///|
pub fn admit_luna_tile_program(
  input : LunaTileProgramInput,
  limits : LunaTileLimits,
) -> LunaTileProgram raise LunaTileError {
  if input.tensors.length() == 0 || input.tensors.length() > limits.max_tensors {
    raise LimitExceeded(TensorCount)
  }
  if input.instructions.length() == 0 ||
    input.instructions.length() > limits.max_instructions {
    raise LimitExceeded(InstructionCount)
  }
  if input.loop_extents.length() == 0 ||
    input.loop_extents.length() > limits.max_loop_rank {
    raise LimitExceeded(LoopRank)
  }
  if input.constraints.length() > limits.max_constraints {
    raise LimitExceeded(ConstraintCount)
  }
  for extent in input.loop_extents {
    if extent <= 0L {
      raise InvalidProgram(AffineExpression)
    }
  }
  let mut previous_id = 0
  let mut shared_memory_bytes = 0L
  for tensor_index, tensor in input.tensors {
    if tensor.id.as_int() <= previous_id {
      raise InvalidProgram(TensorOrder)
    }
    previous_id = tensor.id.as_int()
    validate_luna_tile_tensor(tensor, limits)
    if tensor.address_space == Shared {
      let tensor_end = luna_tile_tensor_end(tensor)
      for previous_index in 0..<tensor_index {
        let previous = input.tensors[previous_index]
        if previous.address_space == Shared {
          let previous_end = luna_tile_tensor_end(previous)
          if tensor.offset_bytes < previous_end &&
            previous.offset_bytes < tensor_end {
            raise InvalidTensor(tensor.id, TensorOverlap)
          }
        }
      }
      shared_memory_bytes = shared_memory_bytes.max(tensor_end)
    }
  }
  if shared_memory_bytes > limits.max_shared_memory_bytes {
    raise LimitExceeded(SharedMemoryBytes)
  }
  let constraints = canonical_luna_tile_constraints(input.constraints)
  for constraint in constraints {
    validate_luna_tile_constraint(constraint, input.tensors)
  }
  validate_luna_tile_instructions(
    input.instructions,
    input.tensors,
    input.loop_extents,
  )
  match input.residual_add {
    Some(semantic) =>
      validate_luna_tile_residual_add(
        semantic,
        input.loop_extents,
        input.tensors,
        input.instructions,
      )
    None => ()
  }
  let canonical = encode_luna_tile_program(
    input.residual_add,
    input.loop_extents,
    input.tensors,
    input.instructions,
    constraints,
    limits.max_canonical_bytes,
  )
  let digest = LunaTileProgramDigest(
    @crypto.bytes_to_hex_string(@crypto.sha256(canonical)),
  )
  {
    residual_add: input.residual_add,
    loop_extents: input.loop_extents,
    tensors: input.tensors,
    instructions: input.instructions,
    constraints,
    canonical,
    digest,
  }
}
