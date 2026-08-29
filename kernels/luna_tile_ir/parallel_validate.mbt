///|
fn luna_tile_parallel_output_is_disjoint(
  program : LunaTileProgram,
  region : LunaTileRegion,
) -> Bool {
  let used_dimensions = Array::make(program.loop_extents.length(), false)
  for tensor_dimension in 0..<region.origins.length() {
    let origin = region.origins[tensor_dimension]
    if origin.constant == 0L && origin.terms.length() == 1 {
      let term = origin.terms[0]
      if term.coefficient == region.extents[tensor_dimension] &&
        !used_dimensions[term.loop_dimension] {
        used_dimensions[term.loop_dimension] = true
      }
    }
  }
  for loop_dimension in 0..<program.loop_extents.length() {
    if program.loop_extents[loop_dimension] > 1L &&
      !used_dimensions[loop_dimension] {
      return false
    }
  }
  true
}

///|
fn validate_luna_tile_parallel_program(
  program : LunaTileProgram,
) -> Unit raise LunaTileError {
  let mut saw_stage = false
  let mut saw_async = false
  let mut saw_barrier = false
  let mut prefix_finished = false
  let mut stage = -1
  let mut global_writes = 0
  let global_write_tensors : Array[LunaTileTensorId] = []
  let global_write_regions : Array[LunaTileRegion] = []
  let global_read_tensors : Array[LunaTileTensorId] = []
  let global_read_regions : Array[LunaTileRegion] = []
  for instruction in program.instructions {
    match instruction {
      LunaTilePipelineStage(id) => {
        if saw_stage || prefix_finished {
          raise InvalidProgram(ParallelPipeline)
        }
        saw_stage = true
        stage = id.as_int()
      }
      LunaTileAsyncCopy(source, _, instruction_stage) => {
        if !saw_stage ||
          saw_barrier ||
          prefix_finished ||
          instruction_stage.as_int() != stage {
          raise InvalidProgram(ParallelPipeline)
        }
        saw_async = true
        let tensor = require_luna_tile_tensor(program.tensors, source.tensor)
        if tensor.address_space == Global {
          global_read_tensors.push(source.tensor)
          global_read_regions.push(source)
        }
      }
      LunaTileBarrier(barrier_stage, _) => {
        if !saw_async ||
          saw_barrier ||
          prefix_finished ||
          barrier_stage.as_int() != stage {
          raise InvalidProgram(ParallelPipeline)
        }
        saw_barrier = true
      }
      LunaTileCopy(source, destination) => {
        prefix_finished = true
        let source_tensor = require_luna_tile_tensor(
          program.tensors,
          source.tensor,
        )
        if source_tensor.address_space == Global {
          global_read_tensors.push(source.tensor)
          global_read_regions.push(source)
        }
        let tensor = require_luna_tile_tensor(
          program.tensors,
          destination.tensor,
        )
        if tensor.address_space == Global {
          global_writes += 1
          if !luna_tile_parallel_output_is_disjoint(program, destination) {
            raise InvalidProgram(ParallelMapping)
          }
          for index, existing_tensor in global_write_tensors {
            if existing_tensor == destination.tensor &&
              global_write_regions[index] != destination {
              raise InvalidProgram(ParallelMapping)
            }
          }
          global_write_tensors.push(destination.tensor)
          global_write_regions.push(destination)
        }
      }
      LunaTileMma(..) | LunaTileReduce(_, ..) | LunaTileElementwise(_, ..) =>
        prefix_finished = true
    }
  }
  if !saw_stage || !saw_async || !saw_barrier || global_writes == 0 {
    raise InvalidProgram(ParallelPipeline)
  }
  // CUDA has no grid-wide ordering inside this candidate. A global tensor may
  // therefore be both read and written only through the exact same disjoint
  // per-worker region. A differently mapped read could observe another
  // worker's write regardless of instruction order within either worker.
  for read_index, read_tensor in global_read_tensors {
    for write_index, write_tensor in global_write_tensors {
      if read_tensor == write_tensor &&
        global_read_regions[read_index] != global_write_regions[write_index] {
        raise InvalidProgram(ParallelMapping)
      }
    }
  }
}

///|
fn luna_tile_parallel_placed_program(
  program : LunaTileProgram,
  shared : LunaTileParallelSharedPlan,
) -> LunaTileProgram raise LunaTileError {
  let tensors : Array[LunaTileTensorView] = []
  for tensor in program.tensors {
    if tensor.address_space == Shared {
      let mut offset = -1L
      for placement in shared.placements {
        if placement.tensor == tensor.id {
          offset = placement.offset_bytes
        }
      }
      if offset < 0L {
        raise InvalidProgram(ParallelLifetime)
      }
      tensors.push({ ..tensor, offset_bytes: offset })
    } else {
      tensors.push(tensor)
    }
  }
  { ..program, tensors: ReadOnlyArray::from_array(tensors) }
}
