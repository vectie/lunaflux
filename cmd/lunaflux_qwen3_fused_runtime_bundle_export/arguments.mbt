///|
suberror Qwen3FusedRuntimeBundleExportError {
  InvalidArguments
  InvalidModule
  InvalidRuntime
  WriteFailed
} derive(Debug, Eq)

///|
priv struct ModuleLaunchInput {
  first_operation : Int32
  grid_x : Int32
  grid_y : Int32
  grid_z : Int32
  block_x : Int32
  shared_memory_bytes : Int32
  module_path : String
}

///|
priv struct RuntimeBundleArguments {
  device_target : @catalog.DeviceTarget?
  ingress_selection : IngressSelectionBasis
  ingress_query_tile_rows : Int32?
  ingress_heads_per_cta : Int32?
  blockwise_decode : Bool
  contracting_decode : Bool
  dual_score_decode : Bool
  decode_execution : @luna_fused_artifact_admission.DecodeExecutionDescriptor?
  decode_function_symbol : String?
  step_rotary_cache : Bool
  ingress_row_variant : IngressRowVariantArguments?
  query_metadata : Bool
  model_content : String
  model_plan : String
  residual : ModuleLaunchInput
  ingress : ModuleLaunchInput
  attention_prefill : ModuleLaunchInput
  attention_prefill_wide : ModuleLaunchInput
  attention_prefill_wide_symbol : String
  attention_prefill_partial : ModuleLaunchInput
  attention_prefill_merge : ModuleLaunchInput
  attention_prefill_partitioned8_partial : ModuleLaunchInput
  attention_prefill_partitioned8_merge : ModuleLaunchInput
  attention_read : ModuleLaunchInput
  prefill_query_tile_rows : Int32
  wide_query_tile_rows : Int32
  partitioned_query_tile_rows : Int32
  deep_partitioned_query_tile_rows : Int32
  output_path : String
}

///|
fn lower_sha256(value : StringView) -> Bool {
  if value.length() != 64 {
    return false
  }
  for unit in value {
    if !(unit is ('0'..<='9' | 'a'..<='f')) {
      return false
    }
  }
  true
}

///|
fn positive_int(
  value : String,
) -> Int32 raise Qwen3FusedRuntimeBundleExportError {
  let parsed : Int32 = @string.parse_int(value) catch {
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
) -> Int32 raise Qwen3FusedRuntimeBundleExportError {
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
fn safe_function_symbol(symbol : StringView) -> Bool {
  !symbol.is_empty() &&
  symbol.length() <= 512 &&
  !symbol.contains("\n") &&
  !symbol.contains("\r")
}

///|
fn parse_arguments(
  args : ArrayView[String],
) -> RuntimeBundleArguments raise Qwen3FusedRuntimeBundleExportError {
  let (args, query_metadata) = match args {
    [.. rest, "--query-metadata-v1"] => (rest, true)
    _ => (args, false)
  }
  let (
    model_content,
    model_plan,
    residual_operation,
    residual_grid_x,
    residual_grid_y,
    residual_grid_z,
    residual_block_x,
    residual_shared,
    residual_module,
    ingress_operation,
    ingress_grid_x,
    ingress_grid_y,
    ingress_grid_z,
    ingress_block_x,
    ingress_shared,
    ingress_module,
    prefill_operation,
    prefill_grid_x,
    prefill_grid_y,
    prefill_grid_z,
    prefill_block_x,
    prefill_shared,
    prefill_module,
    prefill_wide_operation,
    prefill_wide_grid_x,
    prefill_wide_grid_y,
    prefill_wide_grid_z,
    prefill_wide_block_x,
    prefill_wide_shared,
    prefill_wide_symbol,
    prefill_wide_module,
    prefill_partial_operation,
    prefill_partial_grid_x,
    prefill_partial_grid_y,
    prefill_partial_grid_z,
    prefill_partial_block_x,
    prefill_partial_shared,
    prefill_partial_module,
    prefill_merge_operation,
    prefill_merge_grid_x,
    prefill_merge_grid_y,
    prefill_merge_grid_z,
    prefill_merge_block_x,
    prefill_merge_shared,
    prefill_merge_module,
    prefill_partitioned8_partial_operation,
    prefill_partitioned8_partial_grid_x,
    prefill_partitioned8_partial_grid_y,
    prefill_partitioned8_partial_grid_z,
    prefill_partitioned8_partial_block_x,
    prefill_partitioned8_partial_shared,
    prefill_partitioned8_partial_module,
    prefill_partitioned8_merge_operation,
    prefill_partitioned8_merge_grid_x,
    prefill_partitioned8_merge_grid_y,
    prefill_partitioned8_merge_grid_z,
    prefill_partitioned8_merge_block_x,
    prefill_partitioned8_merge_shared,
    prefill_partitioned8_merge_module,
    attention_operation,
    attention_grid_x,
    attention_grid_y,
    attention_grid_z,
    attention_block_x,
    attention_shared,
    attention_module,
    prefill_query_tile_rows,
    wide_query_tile_rows,
    partitioned_query_tile_rows,
    deep_partitioned_query_tile_rows,
    output_path,
  ) = match args {
    [
      _,
      model_content,
      model_plan,
      residual_operation,
      residual_grid_x,
      residual_grid_y,
      residual_grid_z,
      residual_block_x,
      residual_shared,
      residual_module,
      ingress_operation,
      ingress_grid_x,
      ingress_grid_y,
      ingress_grid_z,
      ingress_block_x,
      ingress_shared,
      ingress_module,
      prefill_operation,
      prefill_grid_x,
      prefill_grid_y,
      prefill_grid_z,
      prefill_block_x,
      prefill_shared,
      prefill_module,
      prefill_wide_operation,
      prefill_wide_grid_x,
      prefill_wide_grid_y,
      prefill_wide_grid_z,
      prefill_wide_block_x,
      prefill_wide_shared,
      prefill_wide_symbol,
      prefill_wide_module,
      prefill_partial_operation,
      prefill_partial_grid_x,
      prefill_partial_grid_y,
      prefill_partial_grid_z,
      prefill_partial_block_x,
      prefill_partial_shared,
      prefill_partial_module,
      prefill_merge_operation,
      prefill_merge_grid_x,
      prefill_merge_grid_y,
      prefill_merge_grid_z,
      prefill_merge_block_x,
      prefill_merge_shared,
      prefill_merge_module,
      prefill_partitioned8_partial_operation,
      prefill_partitioned8_partial_grid_x,
      prefill_partitioned8_partial_grid_y,
      prefill_partitioned8_partial_grid_z,
      prefill_partitioned8_partial_block_x,
      prefill_partitioned8_partial_shared,
      prefill_partitioned8_partial_module,
      prefill_partitioned8_merge_operation,
      prefill_partitioned8_merge_grid_x,
      prefill_partitioned8_merge_grid_y,
      prefill_partitioned8_merge_grid_z,
      prefill_partitioned8_merge_block_x,
      prefill_partitioned8_merge_shared,
      prefill_partitioned8_merge_module,
      attention_operation,
      attention_grid_x,
      attention_grid_y,
      attention_grid_z,
      attention_block_x,
      attention_shared,
      attention_module,
      prefill_query_tile_rows,
      wide_query_tile_rows,
      partitioned_query_tile_rows,
      deep_partitioned_query_tile_rows,
      output_path,
    ] =>
      (
        model_content, model_plan, residual_operation, residual_grid_x, residual_grid_y,
        residual_grid_z, residual_block_x, residual_shared, residual_module, ingress_operation,
        ingress_grid_x, ingress_grid_y, ingress_grid_z, ingress_block_x, ingress_shared,
        ingress_module, prefill_operation, prefill_grid_x, prefill_grid_y, prefill_grid_z,
        prefill_block_x, prefill_shared, prefill_module, prefill_wide_operation,
        prefill_wide_grid_x, prefill_wide_grid_y, prefill_wide_grid_z, prefill_wide_block_x,
        prefill_wide_shared, prefill_wide_symbol, prefill_wide_module, prefill_partial_operation,
        prefill_partial_grid_x, prefill_partial_grid_y, prefill_partial_grid_z, prefill_partial_block_x,
        prefill_partial_shared, prefill_partial_module, prefill_merge_operation,
        prefill_merge_grid_x, prefill_merge_grid_y, prefill_merge_grid_z, prefill_merge_block_x,
        prefill_merge_shared, prefill_merge_module, prefill_partitioned8_partial_operation,
        prefill_partitioned8_partial_grid_x, prefill_partitioned8_partial_grid_y,
        prefill_partitioned8_partial_grid_z, prefill_partitioned8_partial_block_x,
        prefill_partitioned8_partial_shared, prefill_partitioned8_partial_module,
        prefill_partitioned8_merge_operation, prefill_partitioned8_merge_grid_x,
        prefill_partitioned8_merge_grid_y, prefill_partitioned8_merge_grid_z, prefill_partitioned8_merge_block_x,
        prefill_partitioned8_merge_shared, prefill_partitioned8_merge_module, attention_operation,
        attention_grid_x, attention_grid_y, attention_grid_z, attention_block_x,
        attention_shared, attention_module, prefill_query_tile_rows, wide_query_tile_rows,
        partitioned_query_tile_rows, deep_partitioned_query_tile_rows, output_path,
      )
    _ => raise InvalidArguments
  }
  if !lower_sha256(model_content) ||
    !lower_sha256(model_plan) ||
    !safe_absolute_file(residual_module) ||
    !safe_absolute_file(ingress_module) ||
    !safe_absolute_file(prefill_module) ||
    !safe_absolute_file(prefill_wide_module) ||
    !safe_function_symbol(prefill_wide_symbol) ||
    !safe_absolute_file(prefill_partial_module) ||
    !safe_absolute_file(prefill_merge_module) ||
    !safe_absolute_file(prefill_partitioned8_partial_module) ||
    !safe_absolute_file(prefill_partitioned8_merge_module) ||
    !safe_absolute_file(attention_module) ||
    !safe_absolute_file(output_path) {
    raise InvalidArguments
  }
  let residual : ModuleLaunchInput = {
    first_operation: non_negative_int(residual_operation),
    grid_x: positive_int(residual_grid_x),
    grid_y: positive_int(residual_grid_y),
    grid_z: positive_int(residual_grid_z),
    block_x: positive_int(residual_block_x),
    shared_memory_bytes: non_negative_int(residual_shared),
    module_path: residual_module,
  }
  let ingress : ModuleLaunchInput = {
    first_operation: non_negative_int(ingress_operation),
    grid_x: positive_int(ingress_grid_x),
    grid_y: positive_int(ingress_grid_y),
    grid_z: positive_int(ingress_grid_z),
    block_x: positive_int(ingress_block_x),
    shared_memory_bytes: non_negative_int(ingress_shared),
    module_path: ingress_module,
  }
  let attention_prefill_partial : ModuleLaunchInput = {
    first_operation: non_negative_int(prefill_partial_operation),
    grid_x: positive_int(prefill_partial_grid_x),
    grid_y: positive_int(prefill_partial_grid_y),
    grid_z: positive_int(prefill_partial_grid_z),
    block_x: positive_int(prefill_partial_block_x),
    shared_memory_bytes: non_negative_int(prefill_partial_shared),
    module_path: prefill_partial_module,
  }
  let attention_prefill : ModuleLaunchInput = {
    first_operation: non_negative_int(prefill_operation),
    grid_x: positive_int(prefill_grid_x),
    grid_y: positive_int(prefill_grid_y),
    grid_z: positive_int(prefill_grid_z),
    block_x: positive_int(prefill_block_x),
    shared_memory_bytes: non_negative_int(prefill_shared),
    module_path: prefill_module,
  }
  let attention_prefill_wide : ModuleLaunchInput = {
    first_operation: non_negative_int(prefill_wide_operation),
    grid_x: positive_int(prefill_wide_grid_x),
    grid_y: positive_int(prefill_wide_grid_y),
    grid_z: positive_int(prefill_wide_grid_z),
    block_x: positive_int(prefill_wide_block_x),
    shared_memory_bytes: non_negative_int(prefill_wide_shared),
    module_path: prefill_wide_module,
  }
  let attention_prefill_merge : ModuleLaunchInput = {
    first_operation: non_negative_int(prefill_merge_operation),
    grid_x: positive_int(prefill_merge_grid_x),
    grid_y: positive_int(prefill_merge_grid_y),
    grid_z: positive_int(prefill_merge_grid_z),
    block_x: positive_int(prefill_merge_block_x),
    shared_memory_bytes: non_negative_int(prefill_merge_shared),
    module_path: prefill_merge_module,
  }
  let attention_prefill_partitioned8_partial : ModuleLaunchInput = {
    first_operation: non_negative_int(prefill_partitioned8_partial_operation),
    grid_x: positive_int(prefill_partitioned8_partial_grid_x),
    grid_y: positive_int(prefill_partitioned8_partial_grid_y),
    grid_z: positive_int(prefill_partitioned8_partial_grid_z),
    block_x: positive_int(prefill_partitioned8_partial_block_x),
    shared_memory_bytes: non_negative_int(prefill_partitioned8_partial_shared),
    module_path: prefill_partitioned8_partial_module,
  }
  let attention_prefill_partitioned8_merge : ModuleLaunchInput = {
    first_operation: non_negative_int(prefill_partitioned8_merge_operation),
    grid_x: positive_int(prefill_partitioned8_merge_grid_x),
    grid_y: positive_int(prefill_partitioned8_merge_grid_y),
    grid_z: positive_int(prefill_partitioned8_merge_grid_z),
    block_x: positive_int(prefill_partitioned8_merge_block_x),
    shared_memory_bytes: non_negative_int(prefill_partitioned8_merge_shared),
    module_path: prefill_partitioned8_merge_module,
  }
  let attention_read : ModuleLaunchInput = {
    first_operation: non_negative_int(attention_operation),
    grid_x: positive_int(attention_grid_x),
    grid_y: positive_int(attention_grid_y),
    grid_z: positive_int(attention_grid_z),
    block_x: positive_int(attention_block_x),
    shared_memory_bytes: non_negative_int(attention_shared),
    module_path: attention_module,
  }
  if attention_read.first_operation != ingress.first_operation + 3 {
    raise InvalidArguments
  }
  if attention_prefill.first_operation != attention_read.first_operation ||
    attention_prefill_wide.first_operation != attention_read.first_operation ||
    attention_prefill_partial.first_operation != attention_read.first_operation ||
    attention_prefill_merge.first_operation != attention_read.first_operation ||
    attention_prefill_partitioned8_partial.first_operation !=
    attention_read.first_operation ||
    attention_prefill_partitioned8_merge.first_operation !=
    attention_read.first_operation {
    raise InvalidArguments
  }
  {
    model_content,
    model_plan,
    device_target: None,
    ingress_selection: ConservativeUnmeasured,
    ingress_query_tile_rows: None,
    ingress_heads_per_cta: None,
    blockwise_decode: false,
    contracting_decode: false,
    dual_score_decode: false,
    decode_execution: None,
    decode_function_symbol: None,
    step_rotary_cache: false,
    ingress_row_variant: None,
    residual,
    ingress,
    attention_prefill,
    attention_prefill_wide,
    attention_prefill_wide_symbol: prefill_wide_symbol,
    attention_prefill_partial,
    attention_prefill_merge,
    attention_prefill_partitioned8_partial,
    attention_prefill_partitioned8_merge,
    attention_read,
    prefill_query_tile_rows: positive_int(prefill_query_tile_rows),
    query_metadata,
    wide_query_tile_rows: positive_int(wide_query_tile_rows),
    partitioned_query_tile_rows: positive_int(partitioned_query_tile_rows),
    deep_partitioned_query_tile_rows: positive_int(
      deep_partitioned_query_tile_rows,
    ),
    output_path,
  }
}
