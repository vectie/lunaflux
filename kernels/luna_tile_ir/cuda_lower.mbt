///|
pub struct LunaTileCudaLoweringLimits {
  priv max_source_bytes : Int
}

///|
pub fn LunaTileCudaLoweringLimits::new(
  max_source_bytes~ : Int,
) -> LunaTileCudaLoweringLimits raise LunaTileError {
  if max_source_bytes <= 0 {
    raise InvalidProgram(CudaLowering)
  }
  { max_source_bytes, }
}

///|
pub struct LunaTileCudaSourceDigest(String) derive(Debug, Eq, Hash)

///|
pub fn LunaTileCudaSourceDigest::from_sha256(
  value : String,
) -> LunaTileCudaSourceDigest raise LunaTileError {
  if !luna_tile_is_sha256(value) {
    raise InvalidProgram(CanonicalEncoding)
  }
  LunaTileCudaSourceDigest(value)
}

///|
pub fn LunaTileCudaSourceDigest::as_hex(
  self : LunaTileCudaSourceDigest,
) -> String {
  self.0
}

///|
pub struct LunaTileCudaTranslationDigest(String) derive(Debug, Eq, Hash)

///|
pub fn LunaTileCudaTranslationDigest::from_sha256(
  value : String,
) -> LunaTileCudaTranslationDigest raise LunaTileError {
  if !luna_tile_is_sha256(value) {
    raise InvalidProgram(CanonicalEncoding)
  }
  LunaTileCudaTranslationDigest(value)
}

///|
pub fn LunaTileCudaTranslationDigest::as_hex(
  self : LunaTileCudaTranslationDigest,
) -> String {
  self.0
}

///|
/// Deterministic, inert CUDA translation unit. It owns no compiler, file,
/// process, module, device, manifest, or promotion authority.
pub struct LunaTileCudaTranslationUnit {
  priv entry_point_id : LunaTileEntryPointId
  priv entry_point : String
  priv program_digest : LunaTileProgramDigest
  priv planning_digest : LunaTileCudaAotDigest
  priv memory_plan : LunaTileMemoryPlan
  priv source : Bytes
  priv source_digest : LunaTileCudaSourceDigest
  priv canonical : Bytes
  priv digest : LunaTileCudaTranslationDigest
} derive(Debug, Eq)

///|
pub fn LunaTileCudaTranslationUnit::entry_point_id(
  self : LunaTileCudaTranslationUnit,
) -> LunaTileEntryPointId {
  self.entry_point_id
}

///|
pub fn LunaTileCudaTranslationUnit::entry_point(
  self : LunaTileCudaTranslationUnit,
) -> String {
  self.entry_point
}

///|
pub fn LunaTileCudaTranslationUnit::program_digest(
  self : LunaTileCudaTranslationUnit,
) -> LunaTileProgramDigest {
  self.program_digest
}

///|
pub fn LunaTileCudaTranslationUnit::planning_digest(
  self : LunaTileCudaTranslationUnit,
) -> LunaTileCudaAotDigest {
  self.planning_digest
}

///|
pub fn LunaTileCudaTranslationUnit::memory_plan(
  self : LunaTileCudaTranslationUnit,
) -> LunaTileMemoryPlan {
  self.memory_plan
}

///|
pub fn LunaTileCudaTranslationUnit::source_bytes(
  self : LunaTileCudaTranslationUnit,
) -> Bytes {
  self.source
}

///|
pub fn LunaTileCudaTranslationUnit::source_digest(
  self : LunaTileCudaTranslationUnit,
) -> LunaTileCudaSourceDigest {
  self.source_digest
}

///|
pub fn LunaTileCudaTranslationUnit::canonical_bytes(
  self : LunaTileCudaTranslationUnit,
) -> Bytes {
  self.canonical
}

///|
pub fn LunaTileCudaTranslationUnit::digest(
  self : LunaTileCudaTranslationUnit,
) -> LunaTileCudaTranslationDigest {
  self.digest
}

///|
fn luna_tile_affine_byte_aligned(
  expression : LunaTileAffineExpr,
  stride : Int64,
  width : Int64,
) -> Bool raise LunaTileError {
  if luna_tile_checked_mul(expression.constant, stride, AotPlanningPolicy) %
    width !=
    0L {
    return false
  }
  for term in expression.terms {
    if luna_tile_checked_mul(term.coefficient, stride, AotPlanningPolicy) %
      width !=
      0L {
      return false
    }
  }
  true
}

///|
fn luna_tile_region_supports_vector(
  region : LunaTileRegion,
  tensor : LunaTileTensorView,
  width : Int,
) -> Bool raise LunaTileError {
  let last = tensor.shape.length() - 1
  let width64 = Int64::from_int(width)
  let element_bytes = luna_tile_dtype_bytes(tensor.dtype)
  if tensor.alignment_bytes < width64 ||
    tensor.offset_bytes % width64 != 0L ||
    tensor.strides_bytes[last] != element_bytes ||
    luna_tile_checked_mul(
      region.extents[last],
      element_bytes,
      AotPlanningPolicy,
    ) %
    width64 !=
    0L {
    return false
  }
  for dimension in 0..<tensor.shape.length() {
    if !luna_tile_affine_byte_aligned(
        region.origins[dimension],
        tensor.strides_bytes[dimension],
        width64,
      ) ||
      (
        dimension < last &&
        region.extents[dimension] > 1L &&
        tensor.strides_bytes[dimension] % width64 != 0L
      ) {
      return false
    }
  }
  true
}

///|
fn luna_tile_copy_vector_width(
  source : LunaTileRegion,
  source_tensor : LunaTileTensorView,
  destination : LunaTileRegion,
  destination_tensor : LunaTileTensorView,
  requested : Int,
) -> Int raise LunaTileError {
  let mut width = requested
  while width >= 4 {
    if luna_tile_region_supports_vector(source, source_tensor, width) &&
      luna_tile_region_supports_vector(destination, destination_tensor, width) {
      return width
    }
    width = width / 2
  }
  0
}

///|
fn luna_tile_cuda_type(dtype : LunaTileDType) -> String {
  match dtype {
    BFloat16 => "__nv_bfloat16"
    Float16 => "__half"
    Float32 => "float"
    Int8 => "signed char"
    Int32 => "int"
  }
}

///|
fn luna_tile_cuda_vector_type(width : Int) -> String {
  match width {
    16 => "uint4"
    8 => "uint2"
    4 => "uint32_t"
    2 => "uint16_t"
    _ => "uint8_t"
  }
}

///|
fn luna_tile_cuda_base(tensor : LunaTileTensorView) -> String {
  match tensor.address_space {
    Global => "tensor_" + tensor.id.as_int().to_string()
    Shared => "luna_shared"
    Register => "luna_reg_" + tensor.id.as_int().to_string()
  }
}

///|
fn luna_tile_cuda_affine(expression : LunaTileAffineExpr) -> String {
  let text = StringBuilder()
  text.write_string(expression.constant.to_string())
  for term in expression.terms {
    text.write_string(" + l_")
    text.write_string(term.loop_dimension.to_string())
    text.write_string(" * ")
    text.write_string(term.coefficient.to_string())
  }
  text.to_string()
}

///|
fn luna_tile_cuda_address(
  tensor : LunaTileTensorView,
  indices : ArrayView[String],
) -> String {
  let text = StringBuilder()
  text.write_string(luna_tile_cuda_base(tensor))
  text.write_string(" + ")
  text.write_string(tensor.offset_bytes.to_string())
  for index in 0..<indices.length() {
    text.write_string(" + (")
    text.write_string(indices[index])
    text.write_string(") * ")
    text.write_string(tensor.strides_bytes[index].to_string())
  }
  text.to_string()
}

///|
fn luna_tile_cuda_value(
  tensor : LunaTileTensorView,
  indices : ArrayView[String],
) -> String {
  "(*reinterpret_cast<" +
  luna_tile_cuda_type(tensor.dtype) +
  "*>(" +
  luna_tile_cuda_address(tensor, indices) +
  "))"
}

///|
fn luna_tile_cuda_to_float(value : String, dtype : LunaTileDType) -> String {
  match dtype {
    BFloat16 => "__bfloat162float(" + value + ")"
    Float16 => "__half2float(" + value + ")"
    Float32 => value
    Int8 | Int32 => "static_cast<float>(" + value + ")"
  }
}

///|
fn luna_tile_cuda_from_float(value : String, dtype : LunaTileDType) -> String {
  match dtype {
    BFloat16 => "__float2bfloat16_rn(" + value + ")"
    Float16 => "__float2half_rn(" + value + ")"
    Float32 => value
    Int8 | Int32 =>
      "static_cast<" + luna_tile_cuda_type(dtype) + ">(" + value + ")"
  }
}

///|
fn luna_tile_cuda_open_loops(
  output : StringBuilder,
  prefix : String,
  extents : ReadOnlyArray[Int64],
  rank : Int,
) -> Array[String] {
  let indices : Array[String] = []
  for dimension in 0..<rank {
    let name = prefix + "_" + dimension.to_string()
    indices.push(name)
    output.write_string("    for (long long ")
    output.write_string(name)
    output.write_string(" = 0; ")
    output.write_string(name)
    output.write_string(" < ")
    output.write_string(extents[dimension].to_string())
    output.write_string("LL; ++")
    output.write_string(name)
    output.write_string(") {\n")
  }
  indices
}

///|
fn luna_tile_cuda_close_loops(output : StringBuilder, rank : Int) -> Unit {
  for _ in 0..<rank {
    output.write_string("    }\n")
  }
}

///|
fn luna_tile_cuda_region_indices(
  region : LunaTileRegion,
  local_indices : ArrayView[String],
) -> Array[String] {
  Array::makei(region.origins.length(), dimension => {
    let affine = luna_tile_cuda_affine(region.origins[dimension])
    if dimension < local_indices.length() {
      "(" + affine + ") + " + local_indices[dimension]
    } else {
      affine
    }
  })
}

///|
fn lower_luna_tile_cuda_copy(
  output : StringBuilder,
  program : LunaTileProgram,
  source : LunaTileRegion,
  destination : LunaTileRegion,
  instruction_index : Int,
  requested_vector : Int,
  asynchronous : Bool,
) -> Unit raise LunaTileError {
  let source_tensor = require_luna_tile_tensor(program.tensors, source.tensor)
  let destination_tensor = require_luna_tile_tensor(
    program.tensors,
    destination.tensor,
  )
  let rank = source.extents.length()
  let vector = luna_tile_copy_vector_width(
    source, source_tensor, destination, destination_tensor, requested_vector,
  )
  if asynchronous && vector < 4 {
    raise InvalidProgram(CudaLowering)
  }
  if vector < 4 {
    let indices = luna_tile_cuda_open_loops(
      output,
      "c" + instruction_index.to_string(),
      source.extents,
      rank,
    )
    let source_indices = luna_tile_cuda_region_indices(source, indices)
    let destination_indices = luna_tile_cuda_region_indices(
      destination, indices,
    )
    output.write_string("      ")
    output.write_string(
      luna_tile_cuda_value(destination_tensor, destination_indices),
    )
    output.write_string(" = ")
    output.write_string(luna_tile_cuda_value(source_tensor, source_indices))
    output.write_string(";\n")
    luna_tile_cuda_close_loops(output, rank)
    return
  }
  let width = vector
  let outer = luna_tile_cuda_open_loops(
    output,
    "c" + instruction_index.to_string(),
    source.extents,
    rank - 1,
  )
  let byte_name = "c" + instruction_index.to_string() + "_bytes"
  let last = rank - 1
  let inner_bytes = luna_tile_checked_mul(
    source.extents[last],
    luna_tile_dtype_bytes(source_tensor.dtype),
    CudaLowering,
  )
  output.write_string("    for (long long ")
  output.write_string(byte_name)
  output.write_string(" = 0; ")
  output.write_string(byte_name)
  output.write_string(" < ")
  output.write_string(inner_bytes.to_string())
  output.write_string("LL; ")
  output.write_string(byte_name)
  output.write_string(" += ")
  output.write_string(width.to_string())
  output.write_string(") {\n")
  let source_indices = luna_tile_cuda_region_indices(source, outer)
  let destination_indices = luna_tile_cuda_region_indices(destination, outer)
  let source_address = luna_tile_cuda_address(source_tensor, source_indices) +
    " + " +
    byte_name
  let destination_address = luna_tile_cuda_address(
      destination_tensor, destination_indices,
    ) +
    " + " +
    byte_name
  if asynchronous {
    output.write_string("      __pipeline_memcpy_async(")
    output.write_string(destination_address)
    output.write_string(", ")
    output.write_string(source_address)
    output.write_string(", ")
    output.write_string(width.to_string())
    output.write_string(");\n")
  } else {
    let vector_type = luna_tile_cuda_vector_type(width)
    output.write_string("      *reinterpret_cast<")
    output.write_string(vector_type)
    output.write_string("*>(")
    output.write_string(destination_address)
    output.write_string(") = *reinterpret_cast<const ")
    output.write_string(vector_type)
    output.write_string("*>(")
    output.write_string(source_address)
    output.write_string(");\n")
  }
  output.write_string("    }\n")
  luna_tile_cuda_close_loops(output, rank - 1)
}
