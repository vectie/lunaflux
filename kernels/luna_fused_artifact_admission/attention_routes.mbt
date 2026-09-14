///|
/// Optional offline measurements travel inside the deployment-pinned bundle.
/// Consumers validate and compile them before constructing device executors.
pub fn ApprovedReusableFusedRuntimeBundle::attention_bucket_routes(
  self : ApprovedReusableFusedRuntimeBundle,
) -> Bytes? {
  self.attention_bucket_routes
}

///|
/// Binds measurements to the exact model plan, target and both compiled
/// attention implementations. Device assignment remains deployment-owned.
pub fn ApprovedReusableFusedRuntimeBundle::attention_route_scope(
  self : ApprovedReusableFusedRuntimeBundle,
) -> String raise FusedArtifactAdmissionError {
  guard self.qwen_paged_attention_prefill_module() is Some(base) &&
    self.qwen_wide_query_prefill_module() is Some(wide) else {
    raise InvalidFusedArtifact(Binding)
  }
  ignore(base)
  ignore(wide)
  let out = StringBuilder()
  out.write_string(
    "attention-chain-scope-v1\n" +
    self.model_content.as_hex() +
    "\n" +
    self.model_plan.as_hex() +
    "\n",
  )
  reusable_runtime_append_target(out, self.target)
  for module_ in self.modules {
    out.write_string(
      reusable_runtime_abi_name(module_.abi) +
      ":" +
      module_.module_digest.as_hex() +
      ":" +
      module_.symbol +
      "\n",
    )
    let launch = module_.launch
    for
      number in [
        module_.first_operation,
        module_.operation_count,
        launch.grid_x(),
        launch.grid_y(),
        launch.grid_z(),
        launch.block_x(),
        launch.block_y(),
        launch.block_z(),
        launch.shared_memory_bytes(),
        module_.query_tile_rows.unwrap_or(0),
        module_.query_metadata_version,
      ] {
      out.write_string(number.to_string() + ":")
    }
    out.write_string("\n")
  }
  @crypto.bytes_to_hex_string(@crypto.sha256(@utf8.encode(out.to_string())))
}
