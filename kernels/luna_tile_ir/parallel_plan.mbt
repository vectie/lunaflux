///|
priv struct LunaTileSharedLifetime {
  tensor : LunaTileTensorView
  first : Int
  last : Int
  size : Int64
}

///|
priv struct LunaTileSharedSlot {
  offset : Int64
  capacity : Int64
  mut last_use : Int
}

///|
fn luna_tile_instruction_touches(
  instruction : LunaTileInstruction,
  tensor : LunaTileTensorId,
) -> Bool {
  match instruction {
    LunaTileCopy(source, destination)
    | LunaTileAsyncCopy(source, destination, _) =>
      source.tensor == tensor || destination.tensor == tensor
    LunaTileBarrier(_, _) | LunaTilePipelineStage(_) => false
    LunaTileMma(destination~, left~, right~, accumulator~) =>
      destination == tensor ||
      left == tensor ||
      right == tensor ||
      accumulator == tensor
    LunaTileReduce(_, destination~, source~, ..) =>
      destination == tensor || source == tensor
    LunaTileElementwise(_, destination~, inputs~) => {
      if destination == tensor {
        return true
      }
      for input in inputs {
        if input == tensor {
          return true
        }
      }
      false
    }
  }
}

///|
fn luna_tile_shared_lifetimes(
  program : LunaTileProgram,
) -> Array[LunaTileSharedLifetime] raise LunaTileError {
  let lifetimes : Array[LunaTileSharedLifetime] = []
  for tensor in program.tensors {
    if tensor.address_space == Shared {
      let mut first = -1
      let mut last = -1
      for index, instruction in program.instructions {
        if luna_tile_instruction_touches(instruction, tensor.id) {
          if first < 0 {
            first = index
          }
          last = index
        }
      }
      if first < 0 {
        raise InvalidProgram(ParallelLifetime)
      }
      let size = luna_tile_tensor_end(tensor) - tensor.offset_bytes
      let lifetime = { tensor, first, last, size }
      let mut insertion = lifetimes.length()
      while insertion > 0 {
        let previous = lifetimes[insertion - 1]
        if previous.first < first ||
          (
            previous.first == first &&
            previous.tensor.id.as_int() < tensor.id.as_int()
          ) {
          break
        }
        insertion -= 1
      }
      lifetimes.insert(insertion, lifetime)
    }
  }
  lifetimes
}

///|
fn luna_tile_align_up(
  value : Int64,
  alignment : Int64,
) -> Int64 raise LunaTileError {
  let remainder = value % alignment
  if remainder == 0L {
    value
  } else {
    luna_tile_checked_add(value, alignment - remainder, ParallelLifetime)
  }
}

///|
fn plan_luna_tile_parallel_shared(
  program : LunaTileProgram,
  policy : LunaTileParallelPolicy,
) -> LunaTileParallelSharedPlan raise LunaTileError {
  let lifetimes = luna_tile_shared_lifetimes(program)
  let placements : Array[LunaTileSharedPlacement] = []
  let slots : Array[LunaTileSharedSlot] = []
  let mut arena_bytes = 0L
  let mut reuse_count = 0
  for lifetime in lifetimes {
    let alignment = lifetime.tensor.alignment_bytes
    if alignment > 16L {
      raise InvalidProgram(ParallelLifetime)
    }
    let mut selected = -1
    for index, slot in slots {
      if slot.last_use < lifetime.first &&
        slot.capacity >= lifetime.size &&
        slot.offset % alignment == 0L {
        selected = index
        break
      }
    }
    let (offset, reused) = if selected >= 0 {
      slots[selected].last_use = lifetime.last
      reuse_count += 1
      (slots[selected].offset, true)
    } else {
      let offset = luna_tile_align_up(arena_bytes, alignment)
      arena_bytes = luna_tile_checked_add(
        offset,
        lifetime.size,
        ParallelLifetime,
      )
      slots.push({ offset, capacity: lifetime.size, last_use: lifetime.last })
      (offset, false)
    }
    placements.push({
      tensor: lifetime.tensor.id,
      offset_bytes: offset,
      size_bytes: lifetime.size,
      first_instruction: lifetime.first,
      last_instruction: lifetime.last,
      reused_slot: reused,
    })
  }
  arena_bytes = luna_tile_align_up(arena_bytes, 16L)
  let per_block = luna_tile_checked_mul(
    arena_bytes,
    Int64::from_int(policy.pipeline_stages),
    ParallelLifetime,
  )
  let total_shared_memory_bytes = luna_tile_checked_mul(
    per_block,
    Int64::from_int(policy.block_threads),
    ParallelLifetime,
  )
  if total_shared_memory_bytes > policy.max_shared_memory_bytes {
    raise InvalidProgram(ParallelLifetime)
  }
  {
    placements: ReadOnlyArray::from_array(placements),
    arena_bytes,
    total_shared_memory_bytes,
    reuse_count,
  }
}

///|
fn plan_luna_tile_parallel_mapping(
  program : LunaTileProgram,
  policy : LunaTileParallelPolicy,
) -> LunaTileParallelMapping raise LunaTileError {
  let mut saw_global_write = false
  let mut saw_mma = false
  for instruction in program.instructions {
    match instruction {
      LunaTileCopy(_, destination) => {
        let tensor = require_luna_tile_tensor(
          program.tensors,
          destination.tensor,
        )
        if tensor.address_space == Global {
          saw_global_write = true
          let rank = destination.extents.length()
          if rank == 1 {
            if policy.tile_m != 1 ||
              destination.extents[0] != Int64::from_int(policy.tile_n) {
              raise InvalidProgram(ParallelMapping)
            }
          } else if destination.extents[rank - 2] !=
            Int64::from_int(policy.tile_m) ||
            destination.extents[rank - 1] != Int64::from_int(policy.tile_n) {
            raise InvalidProgram(ParallelMapping)
          }
        }
      }
      LunaTileMma(left~, ..) => {
        saw_mma = true
        let tensor = require_luna_tile_tensor(program.tensors, left)
        if tensor.shape[1] != Int64::from_int(policy.tile_k) {
          raise InvalidProgram(ParallelMapping)
        }
      }
      _ => ()
    }
  }
  if !saw_global_write || (!saw_mma && policy.tile_k != 1) {
    raise InvalidProgram(ParallelMapping)
  }
  let mut tile_count = 1L
  for extent in program.loop_extents {
    tile_count = luna_tile_checked_mul(tile_count, extent, ParallelMapping)
  }
  let workers = luna_tile_checked_mul(
    Int64::from_int(policy.block_threads),
    Int64::from_int(policy.block_count),
    ParallelMapping,
  )
  if tile_count % workers != 0L {
    raise InvalidProgram(ParallelMapping)
  }
  let rounds_per_worker = tile_count / workers
  if rounds_per_worker < Int64::from_int(policy.pipeline_stages) {
    raise InvalidProgram(ParallelPipeline)
  }
  {
    block_threads: policy.block_threads,
    block_count: policy.block_count,
    warps_per_block: policy.block_threads / 32,
    tile_m: policy.tile_m,
    tile_n: policy.tile_n,
    tile_k: policy.tile_k,
    tile_count,
    rounds_per_worker,
  }
}

///|
fn luna_tile_tensor_core_eligibility(
  program : LunaTileProgram,
  policy : LunaTileParallelPolicy,
) -> LunaTileTensorCoreEligibility raise LunaTileError {
  let mut mma_count = 0
  let mut shape_eligible = true
  for instruction in program.instructions {
    match instruction {
      LunaTileMma(destination~, left~, right~, ..) => {
        mma_count += 1
        let output = require_luna_tile_tensor(program.tensors, destination)
        let lhs = require_luna_tile_tensor(program.tensors, left)
        let rhs = require_luna_tile_tensor(program.tensors, right)
        if output.shape[0] != Int64::from_int(policy.tile_m) ||
          output.shape[1] != Int64::from_int(policy.tile_n) ||
          lhs.shape[1] != Int64::from_int(policy.tile_k) ||
          policy.tile_m % 16 != 0 ||
          policy.tile_n % 16 != 0 ||
          policy.tile_k % 16 != 0 ||
          lhs.alignment_bytes < 16L ||
          rhs.alignment_bytes < 16L {
          shape_eligible = false
        }
      }
      _ => ()
    }
  }
  if mma_count == 0 {
    TensorCoreNoMma
  } else if policy.compute_major < 8 {
    TensorCoreIneligibleTarget
  } else if !shape_eligible {
    TensorCoreIneligibleShape
  } else {
    TensorCoreEligibleUnqualified
  }
}
