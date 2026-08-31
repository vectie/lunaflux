///|
suberror Qwen3FusedResidualRuntimeExportError {
  InvalidArguments
  InvalidModule
  InvalidRuntime
  WriteFailed
}

///|
fn lower_sha256(value : StringView) -> Bool {
  if value.length() != 64 {
    return false
  }
  for unit in value {
    if !(unit is ('0'..='9' | 'a'..='f')) {
      return false
    }
  }
  true
}

///|
fn positive_int(
  value : String,
) -> Int raise Qwen3FusedResidualRuntimeExportError {
  let parsed : Int = @strconv.from_str(value) catch {
    _ => raise InvalidArguments
  }
  if parsed <= 0 || parsed.to_string() != value {
    raise InvalidArguments
  }
  parsed
}

///|
async fn run(
  args : ArrayView[String],
) -> Unit raise Qwen3FusedResidualRuntimeExportError {
  guard args
    is [
      _,
      model_content,
      model_plan,
      residual_text,
      norm_text,
      grid_text,
      module_path,
      output_path,
    ] else {
    raise InvalidArguments
  }
  if !lower_sha256(model_content) ||
    !lower_sha256(model_plan) ||
    !module_path.has_prefix("/") ||
    !output_path.has_prefix("/") {
    raise InvalidArguments
  }
  let residual = positive_int(residual_text)
  let norm = positive_int(norm_text)
  let grid = positive_int(grid_text)
  if norm != residual + 1 {
    raise InvalidArguments
  }
  let module_bytes = @fs.read_file(module_path).binary() catch {
    _ => raise InvalidModule
  }
  let target = @catalog.DeviceTarget::new(
    compute_major=12,
    compute_minor=0,
    supports_bf16=true,
    supports_cublas_lt=false,
  ) catch {
    _ => raise InvalidRuntime
  }
  let launch = @launch_contract.AotLaunchDimensions::new(
    grid_x=grid,
    grid_y=1,
    grid_z=1,
    block_x=128,
    block_y=1,
    block_z=1,
    shared_memory_bytes=512,
  ) catch {
    _ => raise InvalidRuntime
  }
  let runtime = @luna_fused_artifact_admission.export_reusable_fused_residual_runtime(
    model_content_sha256=model_content,
    model_plan_sha256=model_plan,
    target~,
    residual_operation=residual,
    norm_operation=norm,
    function_symbol="lunaflux_fused_residual_rmsnorm_bf16_block128_tree_production_v2",
    launch~,
    module_bytes~,
  ) catch {
    _ => raise InvalidRuntime
  }
  @fs.write_file(
    output_path,
    runtime,
    create_mode=CreateNew,
    permission=0o444,
    sync=Full,
  ) catch {
    _ => raise WriteFailed
  }
  println("schema=lunaflux-qwen3-fused-residual-runtime-export.v1")
  println(
    "runtime_sha256=" + @crypto.bytes_to_hex_string(@crypto.sha256(runtime)),
  )
  println(
    "module_sha256=" + @crypto.bytes_to_hex_string(@crypto.sha256(module_bytes)),
  )
  println("residual_operation=" + residual.to_string())
  println("norm_operation=" + norm.to_string())
  println("grid_x=" + grid.to_string())
}

///|
async fn main {
  try! run(@env.args())
}
