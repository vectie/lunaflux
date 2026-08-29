///|
fn tensor_core_static_region(
  region : LunaTileRegion,
  tensor : LunaTileTensorId,
) -> Bool {
  if region.tensor != tensor ||
    region.extents != [16L, 16L] ||
    region.origins.length() != 2 {
    return false
  }
  for origin in region.origins {
    if origin.constant != 0L || origin.terms.length() != 0 {
      return false
    }
  }
  true
}

///|
fn tensor_core_tiled_region(
  region : LunaTileRegion,
  tensor : LunaTileTensorId,
) -> Bool {
  if region.tensor != tensor ||
    region.extents != [16L, 16L] ||
    region.origins.length() != 2 {
    return false
  }
  let row = region.origins[0]
  let column = region.origins[1]
  row.constant == 0L &&
  row.terms.length() == 1 &&
  row.terms[0].loop_dimension == 0 &&
  row.terms[0].coefficient == 16L &&
  column.constant == 0L &&
  column.terms.length() == 0
}

///|
fn tensor_core_exact_tensor(
  tensor : LunaTileTensorView,
  dtype : LunaTileDType,
  space : LunaTileAddressSpace,
  shape : ReadOnlyArray[Int64],
  strides : ReadOnlyArray[Int64],
  offset_bytes : Int64,
  minimum_alignment_bytes : Int64,
) -> Bool {
  tensor.dtype == dtype &&
  tensor.address_space == space &&
  tensor.shape == shape &&
  tensor.strides_bytes == strides &&
  tensor.offset_bytes == offset_bytes &&
  tensor.alignment_bytes >= minimum_alignment_bytes
}

///|
/// Authenticates the exact graph consumed by the initial tensor-core lowering:
/// two tiled row-major BF16 global-to-shared copies, one block barrier, one
/// static F32 accumulator load, one 16x16x16 MMA, and one tiled F32 output.
pub fn admit_luna_tile_bf16_mma16x16x16_view(
  program : LunaTileProgram,
) -> LunaTileBf16Mma16x16x16View raise LunaTileError {
  if program.loop_extents.length() != 1 ||
    program.loop_extents[0] <= 0L ||
    program.loop_extents[0] > 65535L ||
    program.tensors.length() != 8 ||
    program.instructions.length() != 7 {
    raise InvalidProgram(TensorCoreQualification)
  }
  let left_global = program.tensors[0]
  let left_shared = program.tensors[1]
  let right_global = program.tensors[2]
  let right_shared = program.tensors[3]
  let accumulator_global = program.tensors[4]
  let accumulator_register = program.tensors[5]
  let output_register = program.tensors[6]
  let output_global = program.tensors[7]
  let rows = luna_tile_checked_mul(
    program.loop_extents[0],
    16L,
    TensorCoreQualification,
  )
  if !tensor_core_exact_tensor(
      left_global,
      BFloat16,
      Global,
      [rows, 16L],
      [32L, 2L],
      0L,
      32L,
    ) ||
    !tensor_core_exact_tensor(
      left_shared,
      BFloat16,
      Shared,
      [16L, 16L],
      [32L, 2L],
      0L,
      16L,
    ) ||
    !tensor_core_exact_tensor(
      right_global,
      BFloat16,
      Global,
      [rows, 16L],
      [32L, 2L],
      0L,
      32L,
    ) ||
    !tensor_core_exact_tensor(
      right_shared,
      BFloat16,
      Shared,
      [16L, 16L],
      [32L, 2L],
      512L,
      16L,
    ) ||
    !tensor_core_exact_tensor(
      accumulator_global,
      Float32,
      Global,
      [16L, 16L],
      [64L, 4L],
      0L,
      32L,
    ) ||
    !tensor_core_exact_tensor(
      accumulator_register,
      Float32,
      Register,
      [16L, 16L],
      [64L, 4L],
      0L,
      32L,
    ) ||
    !tensor_core_exact_tensor(
      output_register,
      Float32,
      Register,
      [16L, 16L],
      [64L, 4L],
      0L,
      32L,
    ) ||
    !tensor_core_exact_tensor(
      output_global,
      Float32,
      Global,
      [rows, 16L],
      [64L, 4L],
      0L,
      32L,
    ) {
    raise InvalidProgram(TensorCoreQualification)
  }
  guard program.instructions
    is [
      LunaTilePipelineStage(stage),
      LunaTileAsyncCopy(left_source, left_destination, left_stage),
      LunaTileAsyncCopy(right_source, right_destination, right_stage),
      LunaTileBarrier(barrier_stage, Block),
      LunaTileCopy(accumulator_source, accumulator_destination),
      LunaTileMma(destination~, left~, right~, accumulator~),
      LunaTileCopy(output_source, output_destination),
    ] else {
    raise InvalidProgram(TensorCoreQualification)
  }
  if stage != left_stage ||
    stage != right_stage ||
    stage != barrier_stage ||
    !tensor_core_tiled_region(left_source, left_global.id) ||
    !tensor_core_static_region(left_destination, left_shared.id) ||
    !tensor_core_tiled_region(right_source, right_global.id) ||
    !tensor_core_static_region(right_destination, right_shared.id) ||
    !tensor_core_static_region(accumulator_source, accumulator_global.id) ||
    !tensor_core_static_region(accumulator_destination, accumulator_register.id) ||
    destination != output_register.id ||
    left != left_shared.id ||
    right != right_shared.id ||
    accumulator != accumulator_register.id ||
    !tensor_core_static_region(output_source, output_register.id) ||
    !tensor_core_tiled_region(output_destination, output_global.id) {
    raise InvalidProgram(TensorCoreQualification)
  }
  {
    program_digest: program.digest,
    tile_count: program.loop_extents[0].to_int(),
    left_global: left_global.id,
    right_global: right_global.id,
    accumulator_global: accumulator_global.id,
    output_global: output_global.id,
  }
}
