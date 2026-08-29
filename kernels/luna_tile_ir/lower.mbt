///|
pub struct LunaTileAotPlanningPolicy {
  priv vector_bytes : Int
  priv warp_width : Int
  priv pipeline_stages : Int
  priv max_shared_memory_bytes : Int64
} derive(Debug, Eq)

///|
pub fn LunaTileAotPlanningPolicy::new(
  vector_bytes~ : Int,
  warp_width~ : Int,
  pipeline_stages~ : Int,
  max_shared_memory_bytes~ : Int64,
) -> LunaTileAotPlanningPolicy raise LunaTileError {
  if !(vector_bytes == 4 || vector_bytes == 8 || vector_bytes == 16) ||
    warp_width != 32 ||
    pipeline_stages <= 0 ||
    pipeline_stages > 8 ||
    max_shared_memory_bytes < 0L {
    raise InvalidProgram(AotPlanningPolicy)
  }
  { vector_bytes, warp_width, pipeline_stages, max_shared_memory_bytes }
}

///|
pub struct LunaTileMemoryPlan {
  priv shared_memory_bytes : Int64
  priv shared_tensor_count : Int
  priv vector_bytes : Int
  priv pipeline_stages : Int
} derive(Debug, Eq)

///|
pub fn LunaTileMemoryPlan::shared_memory_bytes(
  self : LunaTileMemoryPlan,
) -> Int64 {
  self.shared_memory_bytes
}

///|
pub fn LunaTileMemoryPlan::shared_tensor_count(
  self : LunaTileMemoryPlan,
) -> Int {
  self.shared_tensor_count
}

///|
/// Maximum copy width selected by the offline lowering policy. Individual
/// instructions may deterministically narrow when their admitted alignment or
/// contiguous extent requires it.
pub fn LunaTileMemoryPlan::vector_bytes(self : LunaTileMemoryPlan) -> Int {
  self.vector_bytes
}

///|
pub fn LunaTileMemoryPlan::pipeline_stages(self : LunaTileMemoryPlan) -> Int {
  self.pipeline_stages
}

///|
pub struct LunaTileCudaAotDigest(String) derive(Debug, Eq, Hash)

///|
pub fn LunaTileCudaAotDigest::from_sha256(
  value : String,
) -> LunaTileCudaAotDigest raise LunaTileError {
  if !luna_tile_is_sha256(value) {
    raise InvalidProgram(CanonicalEncoding)
  }
  LunaTileCudaAotDigest(value)
}

///|
pub fn LunaTileCudaAotDigest::as_hex(self : LunaTileCudaAotDigest) -> String {
  self.0
}

///|
pub struct LunaTileCudaAotInput {
  priv program_digest : LunaTileProgramDigest
  priv memory_plan : LunaTileMemoryPlan
  priv canonical : Bytes
  priv digest : LunaTileCudaAotDigest
} derive(Debug, Eq)

///|
pub fn LunaTileCudaAotInput::program_digest(
  self : LunaTileCudaAotInput,
) -> LunaTileProgramDigest {
  self.program_digest
}

///|
priv enum LunaTileDecodedSemanticPlan {
  LunaTileDecodedNeutral
  LunaTileDecodedResidualAdd(
    left~ : LunaTileTensorId,
    right~ : LunaTileTensorId,
    output~ : LunaTileTensorId,
    width~ : Int64
  )
}

///|
priv struct LunaTileDecodedAotPlan {
  semantic : LunaTileDecodedSemanticPlan
}

///|
pub fn LunaTileCudaAotInput::memory_plan(
  self : LunaTileCudaAotInput,
) -> LunaTileMemoryPlan {
  self.memory_plan
}

///|
pub fn LunaTileCudaAotInput::canonical_bytes(
  self : LunaTileCudaAotInput,
) -> Bytes {
  self.canonical
}

///|
pub fn LunaTileCudaAotInput::digest(
  self : LunaTileCudaAotInput,
) -> LunaTileCudaAotDigest {
  self.digest
}

///|
fn luna_tile_tensor_end(
  tensor : LunaTileTensorView,
) -> Int64 raise LunaTileError {
  let mut maximum = tensor.offset_bytes
  for index in 0..<tensor.shape.length() {
    maximum = luna_tile_checked_add(
      maximum,
      luna_tile_checked_mul(
        tensor.shape[index] - 1L,
        tensor.strides_bytes[index],
        TensorExtent,
      ),
      TensorExtent,
    )
  }
  luna_tile_checked_add(
    maximum,
    luna_tile_dtype_bytes(tensor.dtype),
    TensorExtent,
  )
}

///|
pub fn plan_luna_tile_cuda_aot_input(
  program : LunaTileProgram,
  policy : LunaTileAotPlanningPolicy,
) -> LunaTileCudaAotInput raise LunaTileError {
  let mut shared_memory_bytes = 0L
  let mut shared_tensor_count = 0
  let mut maximum_stage = 0
  for tensor in program.tensors {
    if tensor.address_space == Shared {
      shared_tensor_count += 1
      shared_memory_bytes = shared_memory_bytes.max(
        luna_tile_tensor_end(tensor),
      )
    }
  }
  for instruction in program.instructions {
    match instruction {
      LunaTilePipelineStage(stage) =>
        maximum_stage = maximum_stage.max(stage.as_int())
      LunaTileAsyncCopy(source, destination, _) => {
        let source_tensor = require_luna_tile_tensor(
          program.tensors,
          source.tensor,
        )
        let destination_tensor = require_luna_tile_tensor(
          program.tensors,
          destination.tensor,
        )
        if luna_tile_copy_vector_width(
            source,
            source_tensor,
            destination,
            destination_tensor,
            policy.vector_bytes,
          ) <
          4 {
          raise InvalidProgram(AotPlanningPolicy)
        }
      }
      _ => ()
    }
  }
  if shared_memory_bytes > policy.max_shared_memory_bytes ||
    maximum_stage > policy.pipeline_stages {
    raise InvalidProgram(AotPlanningPolicy)
  }
  let memory_plan : LunaTileMemoryPlan = {
    shared_memory_bytes,
    shared_tensor_count,
    vector_bytes: policy.vector_bytes,
    pipeline_stages: maximum_stage,
  }
  let maximum = program.canonical.length() + 256
  let encoder : LunaTileEncoder = { bytes: [], maximum }
  for byte in @utf8.encode("LUNATILE-CUDA-AOT-PLAN-V2") {
    encoder.append_byte(byte)
  }
  encoder.append_int(policy.vector_bytes)
  encoder.append_int(policy.warp_width)
  encoder.append_int(policy.pipeline_stages)
  encoder.append_i64(shared_memory_bytes)
  encoder.append_int(shared_tensor_count)
  encoder.append_int(maximum_stage)
  match program.residual_add {
    None => {
      encoder.append_int(0)
      encoder.append_int(0)
      encoder.append_int(0)
      encoder.append_int(0)
      encoder.append_i64(0L)
    }
    Some(semantic) => {
      encoder.append_int(1)
      encoder.append_int(semantic.left.as_int())
      encoder.append_int(semantic.right.as_int())
      encoder.append_int(semantic.output.as_int())
      encoder.append_i64(semantic.width)
    }
  }
  encoder.append_int(program.canonical.length())
  for byte in program.canonical {
    encoder.append_byte(byte)
  }
  let canonical = Bytes::from_array(encoder.bytes)
  let digest = LunaTileCudaAotDigest(
    @crypto.bytes_to_hex_string(@crypto.sha256(canonical)),
  )
  let input = { program_digest: program.digest, memory_plan, canonical, digest }
  validate_luna_tile_decoded_semantics(
    program,
    decode_luna_tile_cuda_aot_plan(input),
  )
  input
}

///|
fn validate_luna_tile_decoded_semantics(
  program : LunaTileProgram,
  decoded : LunaTileDecodedAotPlan,
) -> Unit raise LunaTileError {
  match (program.residual_add, decoded.semantic) {
    (None, LunaTileDecodedNeutral) => ()
    (Some(expected), LunaTileDecodedResidualAdd(left~, right~, output~, width~)) =>
      if left != expected.left ||
        right != expected.right ||
        output != expected.output ||
        width != expected.width {
        raise InvalidProgram(CanonicalEncoding)
      }
    _ => raise InvalidProgram(CanonicalEncoding)
  }
}

///|
fn luna_tile_aot_read_i64(
  bytes : Bytes,
  offset : Int,
) -> Int64 raise LunaTileError {
  if offset < 0 || offset > bytes.length() - 8 {
    raise InvalidProgram(CanonicalEncoding)
  }
  let mut result = 0UL
  for index in 0..<8 {
    result = result | (bytes[offset + index].to_uint64() << (index * 8))
  }
  result.reinterpret_as_int64()
}

///|
fn luna_tile_aot_read_int(
  bytes : Bytes,
  offset : Int,
) -> Int raise LunaTileError {
  let value = luna_tile_aot_read_i64(bytes, offset)
  if value < 0L || value > 2147483647L {
    raise InvalidProgram(CanonicalEncoding)
  }
  value.to_int()
}

///|
fn decode_luna_tile_cuda_aot_plan(
  input : LunaTileCudaAotInput,
) -> LunaTileDecodedAotPlan raise LunaTileError {
  let bytes = input.canonical
  if @crypto.bytes_to_hex_string(@crypto.sha256(bytes)) != input.digest.as_hex() {
    raise InvalidProgram(CanonicalEncoding)
  }
  let header = @utf8.encode("LUNATILE-CUDA-AOT-PLAN-V2")
  let scalar_count = 12
  if bytes.length() < header.length() + scalar_count * 8 {
    raise InvalidProgram(CanonicalEncoding)
  }
  for index, expected in header {
    if bytes[index] != expected {
      raise InvalidProgram(CanonicalEncoding)
    }
  }
  let base = header.length()
  let vector_bytes = luna_tile_aot_read_int(bytes, base)
  let warp_width = luna_tile_aot_read_int(bytes, base + 8)
  let policy_stages = luna_tile_aot_read_int(bytes, base + 16)
  let shared_memory_bytes = luna_tile_aot_read_i64(bytes, base + 24)
  let shared_tensor_count = luna_tile_aot_read_int(bytes, base + 32)
  let planned_stages = luna_tile_aot_read_int(bytes, base + 40)
  if vector_bytes != input.memory_plan.vector_bytes ||
    !(vector_bytes == 4 || vector_bytes == 8 || vector_bytes == 16) ||
    warp_width != 32 ||
    policy_stages <= 0 ||
    policy_stages > 8 ||
    shared_memory_bytes != input.memory_plan.shared_memory_bytes ||
    shared_memory_bytes < 0L ||
    shared_tensor_count != input.memory_plan.shared_tensor_count ||
    planned_stages != input.memory_plan.pipeline_stages ||
    planned_stages > policy_stages {
    raise InvalidProgram(CanonicalEncoding)
  }
  let semantic_code = luna_tile_aot_read_int(bytes, base + 48)
  let left = luna_tile_aot_read_int(bytes, base + 56)
  let right = luna_tile_aot_read_int(bytes, base + 64)
  let output = luna_tile_aot_read_int(bytes, base + 72)
  let width = luna_tile_aot_read_i64(bytes, base + 80)
  let program_length = luna_tile_aot_read_int(bytes, base + 88)
  let program_offset = base + scalar_count * 8
  if program_length <= 0 || program_length != bytes.length() - program_offset {
    raise InvalidProgram(CanonicalEncoding)
  }
  let encoded_program = Bytes::makei(program_length, index => {
    bytes[program_offset + index]
  })
  let found_program_digest = @crypto.bytes_to_hex_string(
    @crypto.sha256(encoded_program),
  )
  if found_program_digest != input.program_digest.as_hex() {
    raise InvalidProgram(CanonicalEncoding)
  }
  let semantic = match semantic_code {
    0 => {
      if left != 0 || right != 0 || output != 0 || width != 0L {
        raise InvalidProgram(CanonicalEncoding)
      }
      LunaTileDecodedNeutral
    }
    1 => {
      if left <= 0 ||
        right <= 0 ||
        output <= 0 ||
        left == right ||
        left == output ||
        right == output ||
        width <= 0L {
        raise InvalidProgram(CanonicalEncoding)
      }
      LunaTileDecodedResidualAdd(
        left=LunaTileTensorId(left),
        right=LunaTileTensorId(right),
        output=LunaTileTensorId(output),
        width~,
      )
    }
    _ => raise InvalidProgram(CanonicalEncoding)
  }
  { semantic, }
}
