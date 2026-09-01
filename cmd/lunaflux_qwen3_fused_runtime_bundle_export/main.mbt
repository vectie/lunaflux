///|
suberror Qwen3FusedRuntimeBundleExportError {
  InvalidArguments
  InvalidModule
  InvalidRuntime
  WriteFailed
} derive(Debug, Eq)

///|
priv struct ModuleLaunchInput {
  first_operation : Int
  grid_x : Int
  grid_y : Int
  block_x : Int
  shared_memory_bytes : Int
  module_path : String
}

///|
priv struct RuntimeBundleArguments {
  model_content : String
  model_plan : String
  residual : ModuleLaunchInput
  ingress : ModuleLaunchInput
  attention_read : ModuleLaunchInput
  output_path : String
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
fn positive_int(value : String) -> Int raise Qwen3FusedRuntimeBundleExportError {
  let parsed : Int = @strconv.from_str(value) catch {
    _ => raise InvalidArguments
  }
  if parsed <= 0 || parsed.to_string() != value {
    raise InvalidArguments
  }
  parsed
}

///|
fn non_negative_int(
  value : String,
) -> Int raise Qwen3FusedRuntimeBundleExportError {
  if value == "0" {
    return 0
  }
  positive_int(value)
}

///|
fn safe_absolute_file(path : String) -> Bool {
  path.length() > 1 &&
  path.length() <= 4096 &&
  path.has_prefix("/") &&
  !path.contains("//") &&
  !path.contains("/./") &&
  !path.contains("/../") &&
  !path.has_suffix("/.") &&
  !path.has_suffix("/..")
}

///|
fn parse_arguments(
  args : ArrayView[String],
) -> RuntimeBundleArguments raise Qwen3FusedRuntimeBundleExportError {
  guard args
    is [
      _,
      model_content,
      model_plan,
      residual_operation,
      residual_grid_x,
      residual_grid_y,
      residual_block_x,
      residual_shared,
      residual_module,
      ingress_operation,
      ingress_grid_x,
      ingress_grid_y,
      ingress_block_x,
      ingress_shared,
      ingress_module,
      attention_operation,
      attention_grid_x,
      attention_grid_y,
      attention_block_x,
      attention_shared,
      attention_module,
      output_path,
    ] else {
    raise InvalidArguments
  }
  if !lower_sha256(model_content) ||
    !lower_sha256(model_plan) ||
    !safe_absolute_file(residual_module) ||
    !safe_absolute_file(ingress_module) ||
    !safe_absolute_file(attention_module) ||
    !safe_absolute_file(output_path) {
    raise InvalidArguments
  }
  let residual : ModuleLaunchInput = {
    first_operation: non_negative_int(residual_operation),
    grid_x: positive_int(residual_grid_x),
    grid_y: positive_int(residual_grid_y),
    block_x: positive_int(residual_block_x),
    shared_memory_bytes: non_negative_int(residual_shared),
    module_path: residual_module,
  }
  let ingress : ModuleLaunchInput = {
    first_operation: non_negative_int(ingress_operation),
    grid_x: positive_int(ingress_grid_x),
    grid_y: positive_int(ingress_grid_y),
    block_x: positive_int(ingress_block_x),
    shared_memory_bytes: non_negative_int(ingress_shared),
    module_path: ingress_module,
  }
  let attention_read : ModuleLaunchInput = {
    first_operation: non_negative_int(attention_operation),
    grid_x: positive_int(attention_grid_x),
    grid_y: positive_int(attention_grid_y),
    block_x: positive_int(attention_block_x),
    shared_memory_bytes: non_negative_int(attention_shared),
    module_path: attention_module,
  }
  if attention_read.first_operation != ingress.first_operation + 3 {
    raise InvalidArguments
  }
  { model_content, model_plan, residual, ingress, attention_read, output_path }
}

///|
fn launch_dimensions(
  input : ModuleLaunchInput,
) -> @launch_contract.AotLaunchDimensions raise Qwen3FusedRuntimeBundleExportError {
  @launch_contract.AotLaunchDimensions::new(
    grid_x=input.grid_x,
    grid_y=input.grid_y,
    grid_z=1,
    block_x=input.block_x,
    block_y=1,
    block_z=1,
    shared_memory_bytes=input.shared_memory_bytes,
  ) catch {
    _ => raise InvalidRuntime
  }
}

///|
async fn read_module(
  path : String,
) -> Bytes raise Qwen3FusedRuntimeBundleExportError {
  let bytes = @fs.read_file(path).binary() catch { _ => raise InvalidModule }
  if bytes.is_empty() || bytes.length() > 8388608 {
    raise InvalidModule
  }
  bytes
}

///|
async fn run(
  raw_args : ArrayView[String],
) -> Unit raise Qwen3FusedRuntimeBundleExportError {
  let args = parse_arguments(raw_args)
  let residual_bytes = read_module(args.residual.module_path)
  let ingress_bytes = read_module(args.ingress.module_path)
  let attention_bytes = read_module(args.attention_read.module_path)
  let target = @catalog.DeviceTarget::new(
    compute_major=12,
    compute_minor=0,
    supports_bf16=true,
    supports_cublas_lt=false,
  ) catch {
    _ => raise InvalidRuntime
  }
  let modules : Array[
    @luna_fused_artifact_admission.ReusableFusedRuntimeModuleInput,
  ] = [
    @luna_fused_artifact_admission.ReusableFusedRuntimeModuleInput::new(
      abi=ResidualRmsNormProductionV2,
      first_operation=args.residual.first_operation,
      operation_count=2,
      function_symbol="lunaflux_fused_residual_rmsnorm_bf16_block128_tree_production_v2",
      launch=launch_dimensions(args.residual),
      module_bytes=residual_bytes,
    ),
    @luna_fused_artifact_admission.ReusableFusedRuntimeModuleInput::new(
      abi=QwenQkNormPositionedRopePagedKvWriteProductionV2,
      first_operation=args.ingress.first_operation + 1,
      operation_count=3,
      function_symbol="lunaflux_qwen_qknorm_rope_kvwrite_bf16_head128_production_v2",
      launch=launch_dimensions(args.ingress),
      module_bytes=ingress_bytes,
    ),
    @luna_fused_artifact_admission.ReusableFusedRuntimeModuleInput::new(
      abi=QwenPagedAttentionReadOnlyProductionV2,
      first_operation=args.attention_read.first_operation,
      operation_count=1,
      function_symbol="lunaflux_paged_attention_bf16_readonly_production_v2_ep_3902",
      launch=launch_dimensions(args.attention_read),
      module_bytes=attention_bytes,
    ),
  ] catch {
    _ => raise InvalidRuntime
  }
  let runtime = @luna_fused_artifact_admission.export_reusable_fused_runtime_bundle(
    model_content_sha256=args.model_content,
    model_plan_sha256=args.model_plan,
    target~,
    modules~,
  ) catch {
    _ => raise InvalidRuntime
  }
  @fs.write_file(
    args.output_path,
    runtime,
    create_mode=CreateNew,
    permission=0o444,
    sync=Full,
  ) catch {
    _ => raise WriteFailed
  }
  println("schema=lunaflux-qwen3-fused-runtime-bundle-export.v3")
  println(
    "runtime_sha256=" + @crypto.bytes_to_hex_string(@crypto.sha256(runtime)),
  )
  println(
    "residual_module_sha256=" +
    @crypto.bytes_to_hex_string(@crypto.sha256(residual_bytes)),
  )
  println(
    "ingress_module_sha256=" +
    @crypto.bytes_to_hex_string(@crypto.sha256(ingress_bytes)),
  )
  println(
    "attention_module_sha256=" +
    @crypto.bytes_to_hex_string(@crypto.sha256(attention_bytes)),
  )
}

///|
async fn main {
  try! run(@env.args())
}
