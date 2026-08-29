///|
fn luna_cuda_aot_checked_elements(
  token_rows : Int64,
  width : Int64,
) -> Int64 raise LunaCudaAotError {
  if token_rows <= 0L || width <= 0L {
    raise Invalid(ExecutionShape)
  }
  if token_rows > 9223372036854775807L / width {
    raise SizeOverflow(ElementCount)
  }
  token_rows * width
}

///|
fn luna_cuda_aot_symbol(id : @catalog.AotKernelEntryPointId) -> String {
  "lunaflux_luna_residual_add_v1_ep_" + id.as_int().to_string()
}

///|
fn luna_cuda_aot_source(symbol : String, element_count : Int64) -> Bytes {
  @utf8.encode(
    "#include <cuda_bf16.h>\n" +
    "extern \"C\" __global__ void " +
    symbol +
    "(const __nv_bfloat16 *left, const __nv_bfloat16 *right, " +
    "__nv_bfloat16 *output) {\n" +
    "  const unsigned long long index = " +
    "(unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;\n" +
    "  if (index < " +
    element_count.to_string() +
    "ULL) {\n" +
    "    output[index] = __hadd(left[index], right[index]);\n" +
    "  }\n" +
    "}\n",
  )
}

///|
fn luna_cuda_aot_dimensions(
  element_count : Int64,
) -> @launch_contract.AotLaunchDimensions raise LunaCudaAotError {
  let threads = 256L
  if element_count > 9223372036854775807L - (threads - 1L) {
    raise SizeOverflow(ElementCount)
  }
  let grid_x = (element_count + threads - 1L) / threads
  if grid_x <= 0L || grid_x > 2147483647L {
    raise LimitExceeded(ExecutionShape)
  }
  @launch_contract.AotLaunchDimensions::new(
    grid_x=grid_x.to_int(),
    grid_y=1,
    grid_z=1,
    block_x=threads.to_int(),
    block_y=1,
    block_z=1,
    shared_memory_bytes=0,
  ) catch {
    _ => raise Invalid(ExecutionShape)
  }
}

///|
fn luna_cuda_bool(value : Bool) -> String {
  if value {
    "true"
  } else {
    "false"
  }
}

///|
fn luna_cuda_aot_recipe(
  target : @catalog.DeviceTarget,
  compiler : @luna_specialization.LunaCompilerPolicy,
  source_digest : LunaCudaSourceDigest,
  function_symbol : String,
  dimensions : @launch_contract.AotLaunchDimensions,
) -> Bytes {
  let identity = compiler.identity()
  @utf8.encode(
    "schema=lunaflux-luna-cuda-aot-recipe-v1\n" +
    "source_sha256=" +
    source_digest.as_hex() +
    "\n" +
    "toolchain_sha256=" +
    identity.toolchain_digest_sha256() +
    "\n" +
    "compiler_version=" +
    identity.major().to_string() +
    "." +
    identity.minor().to_string() +
    "." +
    identity.patch().to_string() +
    "\n" +
    "target=sm_" +
    target.compute_major().to_string() +
    target.compute_minor().to_string() +
    "\n" +
    "output=cubin\n" +
    "optimization=" +
    compiler.optimization_level().to_string() +
    "\n" +
    "fmad=" +
    luna_cuda_bool(compiler.fused_multiply_add()) +
    "\n" +
    "reassociate=false\n" +
    "max_registers=" +
    compiler.max_registers().to_string() +
    "\n" +
    "grid=" +
    dimensions.grid_x().to_string() +
    ",1,1\n" +
    "block=256,1,1\n" +
    "function_symbol=" +
    function_symbol +
    "\n" +
    "abi=pointer,pointer,pointer\n",
  )
}

///|
/// Lowers the single admitted residual-add semantic into deterministic CUDA
/// source and a closed compilation recipe.
///
/// No caller-controlled source fragment, flag, path, or function symbol enters
/// this transformation. Compilation remains an offline release operation.
pub fn lower_luna_residual_add_cuda_aot(
  program : @luna_tile_ir.LunaTileProgram,
  planning : @luna_tile_ir.LunaTileCudaAotInput,
  target : @catalog.DeviceTarget,
  shape : @device_profile.ExactExecutionShape,
  entry_point_id : @catalog.AotKernelEntryPointId,
  compiler : @luna_specialization.LunaCompilerPolicy,
  limits : LunaCudaAotLimits,
) -> LunaCudaAotLowering raise LunaCudaAotError {
  guard program.residual_add_semantics() is Some(semantic) else {
    raise Unsupported(Program)
  }
  if planning.program_digest() != program.digest() {
    raise Mismatch(PlanningInput)
  }
  if !target.supports_bf16() {
    raise Unsupported(DeviceTarget)
  }
  if compiler.reassociate() {
    raise Unsupported(CompilerPolicy)
  }
  let element_count = luna_cuda_aot_checked_elements(
    shape.token_rows(),
    semantic.width(),
  )
  if element_count > 9223372036854775807L / 2L {
    raise SizeOverflow(ElementCount)
  }
  let operand_byte_count = element_count * 2L
  let dimensions = luna_cuda_aot_dimensions(element_count)
  let function_symbol = luna_cuda_aot_symbol(entry_point_id)
  let source = luna_cuda_aot_source(function_symbol, element_count)
  if source.length() > limits.max_source_bytes {
    raise LimitExceeded(Source)
  }
  let source_digest = LunaCudaSourceDigest(luna_cuda_aot_sha256(source))
  let recipe = luna_cuda_aot_recipe(
    target, compiler, source_digest, function_symbol, dimensions,
  )
  if recipe.length() > limits.max_recipe_bytes {
    raise LimitExceeded(Recipe)
  }
  let recipe_digest = LunaCudaRecipeDigest(luna_cuda_aot_sha256(recipe))
  {
    target,
    shape,
    entry_point_id,
    compiler,
    width: semantic.width(),
    element_count,
    operand_byte_count,
    dimensions,
    program_digest: program.digest(),
    planning_digest: planning.digest(),
    function_symbol,
    source,
    source_digest,
    recipe,
    recipe_digest,
  }
}
