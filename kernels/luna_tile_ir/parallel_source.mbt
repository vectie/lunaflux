///|
fn luna_tile_parallel_write_arguments(
  output : StringBuilder,
  program : LunaTileProgram,
) -> Unit {
  let mut count = 0
  for tensor in program.tensors {
    if tensor.address_space == Global {
      if count > 0 {
        output.write_string(", ")
      }
      output.write_string(
        "unsigned char *tensor_" + tensor.id.as_int().to_string(),
      )
      count += 1
    }
  }
  if count == 0 {
    output.write_string("void")
  }
}

///|
fn luna_tile_parallel_write_tile_decode(
  output : StringBuilder,
  loop_extents : ReadOnlyArray[Int64],
  linear_name : String,
) -> Unit {
  output.write_string("      long long luna_remaining = " + linear_name + ";\n")
  let mut dimension = loop_extents.length() - 1
  while dimension >= 0 {
    output.write_string("      long long l_")
    output.write_string(dimension.to_string())
    output.write_string(" = luna_remaining % ")
    output.write_string(loop_extents[dimension].to_string())
    output.write_string("LL; luna_remaining /= ")
    output.write_string(loop_extents[dimension].to_string())
    output.write_string("LL;\n")
    dimension -= 1
  }
}

///|
fn luna_tile_parallel_write_registers(
  output : StringBuilder,
  program : LunaTileProgram,
) -> Unit raise LunaTileError {
  for tensor in program.tensors {
    if tensor.address_space == Register {
      output.write_string("  alignas(16) unsigned char luna_reg_")
      output.write_string(tensor.id.as_int().to_string())
      output.write_string("[")
      output.write_string(luna_tile_tensor_end(tensor).to_string())
      output.write_string("];\n")
    }
  }
}

///|
fn luna_tile_parallel_write_prefetch(
  output : StringBuilder,
  program : LunaTileProgram,
  vector_bytes : Int,
) -> Unit raise LunaTileError {
  for index, instruction in program.instructions {
    match instruction {
      LunaTileAsyncCopy(source, destination, _) =>
        lower_luna_tile_cuda_copy(
          output, program, source, destination, index, vector_bytes, true,
        )
      _ => ()
    }
  }
}

///|
fn luna_tile_parallel_write_compute(
  output : StringBuilder,
  program : LunaTileProgram,
  vector_bytes : Int,
) -> Unit raise LunaTileError {
  for index, instruction in program.instructions {
    match instruction {
      LunaTilePipelineStage(_)
      | LunaTileAsyncCopy(_, _, _)
      | LunaTileBarrier(_, _) => ()
      _ =>
        lower_luna_tile_cuda_instruction(
          output, program, instruction, index, vector_bytes,
        )
    }
  }
}

///|
fn luna_tile_parallel_cuda_source(
  program : LunaTileProgram,
  mapping : LunaTileParallelMapping,
  shared : LunaTileParallelSharedPlan,
  policy : LunaTileParallelPolicy,
  entry_point : String,
) -> Bytes raise LunaTileError {
  let output = StringBuilder()
  output.write_string(
    "#include <cuda_bf16.h>\n#include <cuda_fp16.h>\n" +
    "#include <cuda_pipeline.h>\n#include <math.h>\n#include <stdint.h>\n",
  )
  output.write_string(
    "// inert offline parallel SIMT candidate; no runtime authority\n",
  )
  output.write_string("extern \"C\" __global__ void " + entry_point + "(")
  luna_tile_parallel_write_arguments(output, program)
  output.write_string(") {\n")
  output.write_string("  if (gridDim.x != ")
  output.write_string(mapping.block_count.to_string())
  output.write_string(" || gridDim.y != 1 || gridDim.z != 1 || blockDim.x != ")
  output.write_string(mapping.block_threads.to_string())
  output.write_string(
    " || blockDim.y != 1 || blockDim.z != 1) return;\n" +
    "  extern __shared__ __align__(16) unsigned char luna_shared_block[];\n",
  )
  luna_tile_parallel_write_registers(output, program)
  output.write_string(
    "  const long long luna_warp = threadIdx.x / 32;\n" +
    "  const long long luna_lane = threadIdx.x % 32;\n" +
    "  const long long luna_worker = ((long long)blockIdx.x * " +
    mapping.block_threads.to_string() +
    "LL) + (luna_warp * 32LL) + luna_lane;\n",
  )
  output.write_string("  for (long long luna_pipe = 0; luna_pipe < ")
  output.write_string(
    (mapping.rounds_per_worker + Int64::from_int(policy.pipeline_stages - 1)).to_string(),
  )
  output.write_string("LL; ++luna_pipe) {\n")
  output.write_string("    if (luna_pipe < ")
  output.write_string(mapping.rounds_per_worker.to_string())
  output.write_string("LL) {\n")
  output.write_string(
    "      unsigned char *luna_shared = luna_shared_block + " +
    "(((long long)threadIdx.x * " +
    policy.pipeline_stages.to_string() +
    "LL + (luna_pipe % " +
    policy.pipeline_stages.to_string() +
    "LL)) * " +
    shared.arena_bytes.to_string() +
    "LL);\n" +
    "      long long luna_prefetch_tile = luna_pipe * " +
    (mapping.block_threads * mapping.block_count).to_string() +
    "LL + luna_worker;\n",
  )
  luna_tile_parallel_write_tile_decode(
    output,
    program.loop_extents,
    "luna_prefetch_tile",
  )
  luna_tile_parallel_write_prefetch(output, program, policy.vector_bytes)
  output.write_string("      __pipeline_commit();\n    }\n")
  output.write_string("    if (luna_pipe >= ")
  output.write_string((policy.pipeline_stages - 1).to_string())
  output.write_string("LL) {\n      __pipeline_wait_prior(")
  output.write_string((policy.pipeline_stages - 2).to_string())
  output.write_string(");\n      __syncthreads();\n")
  output.write_string(
    "      long long luna_compute_round = luna_pipe - " +
    (policy.pipeline_stages - 1).to_string() +
    "LL;\n" +
    "      unsigned char *luna_shared = luna_shared_block + " +
    "(((long long)threadIdx.x * " +
    policy.pipeline_stages.to_string() +
    "LL + (luna_compute_round % " +
    policy.pipeline_stages.to_string() +
    "LL)) * " +
    shared.arena_bytes.to_string() +
    "LL);\n" +
    "      long long luna_compute_tile = luna_compute_round * " +
    (mapping.block_threads * mapping.block_count).to_string() +
    "LL + luna_worker;\n",
  )
  luna_tile_parallel_write_tile_decode(
    output,
    program.loop_extents,
    "luna_compute_tile",
  )
  luna_tile_parallel_write_compute(output, program, policy.vector_bytes)
  output.write_string("      __syncthreads();\n    }\n  }\n}\n")
  @utf8.encode(output.to_string())
}
