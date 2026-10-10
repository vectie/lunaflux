///|
priv struct IngressRowVariantArguments {
  maximum_token_rows : Int
  query_tile_rows : Int
  shared_bytes : Int
  module_path : String
}

///|
async fn read_ingress_row_variant(
  args : RuntimeBundleArguments,
  input : IngressRowVariantArguments,
) -> @luna_fused_artifact_admission.ReusableFusedRuntimeRowVariant raise Qwen3FusedRuntimeBundleExportError {
  if input.maximum_token_rows < 2 ||
    input.query_tile_rows <= 0 ||
    input.query_tile_rows > input.maximum_token_rows {
    raise InvalidArguments
  }
  let bytes = read_module(input.module_path)
  @luna_fused_artifact_admission.ReusableFusedRuntimeRowVariant::new(
    maximum_token_rows=input.maximum_token_rows,
    query_tile_rows=input.query_tile_rows,
    function_symbol="lunaflux_fused_qwen_qkv_qknorm_rope_kvwrite_bf16_head128_production_cached_rotary_v3",
    launch=@launch_contract.AotLaunchDimensions::new(
      grid_x=(input.maximum_token_rows - 1) / input.query_tile_rows + 1,
      grid_y=args.ingress.grid_y,
      grid_z=1,
      block_x=args.ingress.block_x,
      block_y=1,
      block_z=1,
      shared_memory_bytes=input.shared_bytes,
    ),
    module_bytes=bytes,
    module_label_sha256=runtime_module_locator_label(input.module_path),
  ) catch {
    _ => raise InvalidRuntime
  }
}
