///|
fn lower_luna_tile_cuda_instruction(
  output : StringBuilder,
  program : LunaTileProgram,
  instruction : LunaTileInstruction,
  instruction_index : Int,
  vector_bytes : Int,
) -> Unit raise LunaTileError {
  match instruction {
    LunaTileCopy(source, destination) =>
      lower_luna_tile_cuda_copy(
        output, program, source, destination, instruction_index, vector_bytes, false,
      )
    LunaTileAsyncCopy(source, destination, _) =>
      lower_luna_tile_cuda_copy(
        output, program, source, destination, instruction_index, vector_bytes, true,
      )
    LunaTilePipelineStage(stage) =>
      output.write_string(
        "    // lunatile.pipeline.stage " + stage.as_int().to_string() + "\n",
      )
    LunaTileBarrier(_, scope) => {
      output.write_string("    __pipeline_commit();\n")
      output.write_string("    __pipeline_wait_prior(0);\n")
      output.write_string(
        match scope {
          Warp => "    __syncwarp();\n"
          Block => "    __syncthreads();\n"
        },
      )
    }
    LunaTileMma(destination~, left~, right~, accumulator~) =>
      lower_luna_tile_cuda_mma(
        output, program, destination, left, right, accumulator, instruction_index,
      )
    LunaTileReduce(kind, destination~, source~, axis~) =>
      lower_luna_tile_cuda_reduce(
        output, program, kind, destination, source, axis, instruction_index,
      )
    LunaTileElementwise(kind, destination~, inputs~) =>
      lower_luna_tile_cuda_elementwise(
        output, program, kind, destination, inputs, instruction_index,
      )
  }
}

///|
fn luna_tile_cuda_translation_source(
  program : LunaTileProgram,
  planning : LunaTileCudaAotInput,
  entry_point : String,
) -> Bytes raise LunaTileError {
  let output = StringBuilder()
  output.write_string(
    "#include <cuda_bf16.h>\n#include <cuda_fp16.h>\n" +
    "#include <cuda_pipeline.h>\n#include <math.h>\n#include <stdint.h>\n",
  )
  output.write_string("extern \"C\" __global__ void " + entry_point + "(")
  let mut argument_count = 0
  for tensor in program.tensors {
    if tensor.address_space == Global {
      if argument_count > 0 {
        output.write_string(", ")
      }
      output.write_string(
        "unsigned char *tensor_" + tensor.id.as_int().to_string(),
      )
      argument_count += 1
    }
  }
  if argument_count == 0 {
    output.write_string("void")
  }
  output.write_string(") {\n")
  output.write_string(
    "  if (gridDim.x != 1 || gridDim.y != 1 || gridDim.z != 1 || " +
    "blockDim.x != 1 || blockDim.y != 1 || blockDim.z != 1) return;\n",
  )
  if planning.memory_plan.shared_tensor_count > 0 {
    output.write_string(
      "  extern __shared__ __align__(16) unsigned char luna_shared[];\n",
    )
  }
  for tensor in program.tensors {
    if tensor.address_space == Register {
      let bytes = luna_tile_tensor_end(tensor)
      output.write_string("  alignas(16) unsigned char luna_reg_")
      output.write_string(tensor.id.as_int().to_string())
      output.write_string("[")
      output.write_string(bytes.to_string())
      output.write_string("];\n")
    }
  }
  ignore(
    luna_tile_cuda_open_loops(
      output,
      "l",
      program.loop_extents,
      program.loop_extents.length(),
    ),
  )
  for index, instruction in program.instructions {
    lower_luna_tile_cuda_instruction(
      output,
      program,
      instruction,
      index,
      planning.memory_plan.vector_bytes,
    )
  }
  luna_tile_cuda_close_loops(output, program.loop_extents.length())
  output.write_string("}\n")
  @utf8.encode(output.to_string())
}

///|
/// Lowers every admitted generic LunaTile instruction to one deterministic,
/// serial-reference CUDA translation unit. The fixed one-block/one-thread
/// launch guard makes the semantics explicit; production parallel launch
/// specialization remains a separately reviewed offline transformation.
pub fn lower_luna_tile_cuda_translation_unit(
  program : LunaTileProgram,
  planning : LunaTileCudaAotInput,
  entry_point_id : LunaTileEntryPointId,
  limits : LunaTileCudaLoweringLimits,
) -> LunaTileCudaTranslationUnit raise LunaTileError {
  if planning.program_digest != program.digest {
    raise InvalidProgram(CudaLowering)
  }
  let entry_point = "lunaflux_lunatile_v1_ep_" +
    entry_point_id.as_int().to_string()
  let source = luna_tile_cuda_translation_source(program, planning, entry_point)
  if source.length() > limits.max_source_bytes {
    raise InvalidProgram(CudaLowering)
  }
  let source_digest = LunaTileCudaSourceDigest(
    @crypto.bytes_to_hex_string(@crypto.sha256(source)),
  )
  let memory_plan = planning.memory_plan
  let canonical = @utf8.encode(
    "schema=lunaflux.lunatile-cuda-translation.v1\n" +
    "entry_point_id=" +
    entry_point_id.as_int().to_string() +
    "\nentry_point=" +
    entry_point +
    "\nprogram_sha256=" +
    program.digest.as_hex() +
    "\nplanning_sha256=" +
    planning.digest.as_hex() +
    "\nsource_sha256=" +
    source_digest.as_hex() +
    "\nsource_bytes=" +
    source.length().to_string() +
    "\nshared_memory_bytes=" +
    memory_plan.shared_memory_bytes.to_string() +
    "\nvector_bytes=" +
    memory_plan.vector_bytes.to_string() +
    "\npipeline_stages=" +
    memory_plan.pipeline_stages.to_string() +
    "\nlaunch=1,1,1;1,1,1\ncompiler_authority=none\n" +
    "runtime_authority=none\nmanifest_authority=none\npromotion_authority=none\n",
  )
  let digest = LunaTileCudaTranslationDigest(
    @crypto.bytes_to_hex_string(@crypto.sha256(canonical)),
  )
  {
    entry_point_id,
    entry_point,
    program_digest: program.digest,
    planning_digest: planning.digest,
    memory_plan,
    source,
    source_digest,
    canonical,
    digest,
  }
}
