///|
pub(all) suberror LoweringError {
  UnsupportedProgram
  UnsupportedTensor
  UnsupportedInstruction
  InvalidBinding
  UninitializedRegister
  Device(@lunachip.AdapterError)
}

///|
/// Base address of a global tensor's allocation, before its view offset.
pub(all) struct GlobalBinding {
  tensor : @tile.LunaTileTensorId
  address : UInt64
}

///|
priv struct Placement {
  tensor : @tile.LunaTileTensorView
  scratch : Int32
  address : UInt64
  count : Int32
  mut initialized : Bool
}

///|
fn lookup(
  places : Array[Placement],
  id : @tile.LunaTileTensorId,
) -> Placement raise LoweringError {
  for place in places {
    if place.tensor.id() == id {
      return place
    }
  }
  raise UnsupportedTensor
}

///|
fn readable(place : Placement) -> Unit raise LoweringError {
  if place.tensor.address_space() == Register && !place.initialized {
    raise UninitializedRegister
  }
}

///|
fn whole_region(
  region : @tile.LunaTileRegion,
  place : Placement,
) -> Unit raise LoweringError {
  let origins = region.origins()
  let extents = region.extents()
  if origins.length() != 1 ||
    extents.length() != 1 ||
    origins[0].constant() != 0L ||
    !origins[0].terms().is_empty() ||
    extents[0] != place.count.to_int64() {
    raise UnsupportedInstruction
  }
}

///|
fn place_tensors(
  device : @lunachip.DeviceContract,
  program : @tile.LunaTileProgram,
  bindings : ArrayView[GlobalBinding],
) -> Array[Placement] raise LoweringError {
  let places : Array[Placement] = []
  let mut next_word = 0
  let mut globals = 0
  for tensor in program.tensors() {
    let shape = tensor.shape()
    let strides = tensor.strides_bytes()
    if tensor.dtype() != Float32 ||
      shape.length() != 1 ||
      shape[0] < 1L ||
      shape[0] > 4096L ||
      strides.length() != 1 ||
      strides[0] != 4L ||
      tensor.alignment_bytes() < 4L {
      raise UnsupportedTensor
    }
    let count = shape[0].to_int()
    match tensor.address_space() {
      Register => {
        if tensor.offset_bytes() != 0L || next_word > 32768 - count {
          raise UnsupportedTensor
        }
        places.push({
          tensor,
          count,
          scratch: next_word,
          address: 0UL,
          initialized: false,
        })
        next_word += count
      }
      Global => {
        globals += 1
        let mut address : UInt64? = None
        for binding in bindings {
          if binding.tensor == tensor.id() {
            if address is Some(_) {
              raise InvalidBinding
            }
            let offset = tensor.offset_bytes().reinterpret_as_uint64()
            device.check_range(binding.address, offset) catch {
              error => raise Device(error)
            }
            address = Some(binding.address + offset)
          }
        }
        guard address is Some(base) else { raise InvalidBinding }
        device.check_range(base, count.to_uint64() * 4UL) catch {
          error => raise Device(error)
        }
        if base % tensor.alignment_bytes().reinterpret_as_uint64() != 0UL {
          raise InvalidBinding
        }
        for previous in places {
          if previous.tensor.address_space() == Global &&
            base < previous.address + previous.count.to_uint64() * 4UL &&
            previous.address < base + count.to_uint64() * 4UL {
            raise InvalidBinding
          }
        }
        places.push({
          tensor,
          count,
          scratch: 0,
          address: base,
          initialized: true,
        })
      }
      _ => raise UnsupportedTensor
    }
  }
  if bindings.length() != globals {
    raise InvalidBinding
  }
  places
}

///|
/// Offline lowering of the closed FP32 copy/add/multiply subset. No fallback,
/// numerical coercion, asynchronous effect erasure or generic MMA lowering.
pub fn lower(
  device : @lunachip.DeviceContract,
  program : @tile.LunaTileProgram,
  bindings : ArrayView[GlobalBinding],
) -> @lunachip.PreparedProgram raise LoweringError {
  if program.loop_rank() != 1 ||
    program.loop_extents()[0] != 1L ||
    program.tensor_count() > 256 ||
    program.instruction_count() > 1048576 ||
    program.residual_add_semantics() is Some(_) {
    raise UnsupportedProgram
  }
  let places = place_tensors(device, program, bindings)
  let operations : Array[@lunachip.Instruction] = []
  for op in program.instructions() {
    let lowered = match op {
      LunaTileCopy(source, destination) => {
        let src = lookup(places, source.tensor())
        let dst = lookup(places, destination.tensor())
        whole_region(source, src)
        whole_region(destination, dst)
        readable(src)
        let result = match
          (src.tensor.address_space(), dst.tensor.address_space()) {
          (Global, Register) =>
            @lunachip.Instruction::load(
              device,
              dst.scratch,
              src.address,
              dst.count,
            ) catch {
              e => raise Device(e)
            }
          (Register, Global) =>
            @lunachip.Instruction::store(
              device,
              src.scratch,
              dst.address,
              src.count,
            ) catch {
              e => raise Device(e)
            }
          (Register, Register) =>
            @lunachip.Instruction::copy(dst.scratch, src.scratch, src.count) catch {
              e => raise Device(e)
            }
          _ => raise UnsupportedInstruction
        }
        dst.initialized = true
        result
      }
      LunaTileElementwise(kind, destination~, inputs~) => {
        if inputs.length() != 2 {
          raise UnsupportedInstruction
        }
        let binary : @lunachip.BinaryOperation = match kind {
          Add => Add
          Multiply => Multiply
          _ => raise UnsupportedInstruction
        }
        let dst = lookup(places, destination)
        let left = lookup(places, inputs[0])
        let right = lookup(places, inputs[1])
        for p in [dst, left, right] {
          if p.tensor.address_space() != Register || p.count != dst.count {
            raise UnsupportedInstruction
          }
        }
        readable(left)
        readable(right)
        let result = @lunachip.Instruction::binary(
          binary,
          dst.scratch,
          left.scratch,
          right.scratch,
          dst.count,
        ) catch {
          e => raise Device(e)
        }
        dst.initialized = true
        result
      }
      _ => raise UnsupportedInstruction
    }
    operations.push(lowered)
  }
  @lunachip.PreparedProgram::new(operations[:]) catch {
    error => raise Device(error)
  }
}
