///|
/// Optional offline measurements travel inside the deployment-pinned bundle.
/// Consumers validate and compile them before constructing device executors.
pub fn ApprovedReusableFusedRuntimeBundle::attention_bucket_routes(
  self : ApprovedReusableFusedRuntimeBundle,
) -> Bytes? {
  self.attention_bucket_routes
}

///|
/// Computes an offline measurement scope from already-validated typed inputs.
/// No serialization, CUBIN hashing or decoding of a generated bundle occurs.
pub fn reusable_fused_runtime_attention_route_scope(
  model_content_sha256~ : String,
  model_plan_sha256~ : String,
  target~ : @catalog.DeviceTarget,
  modules~ : ArrayView[ReusableFusedRuntimeModuleInput],
) -> String raise FusedArtifactAdmissionError {
  let model_content = fused_deployment_digest(model_content_sha256)
  let model_plan = fused_deployment_digest(model_plan_sha256)
  let ordered = modules.to_array()
  ordered.sort_by((left, right) => {
    reusable_runtime_abi_rank(left.abi).compare(
      reusable_runtime_abi_rank(right.abi),
    )
  })
  let metadata = ReadOnlyArray::from_array(
    ordered.map(m => {
      abi: m.abi,
      first_operation: m.first_operation,
      operation_count: m.operation_count,
      symbol: m.symbol,
      launch: m.launch,
      query_tile_rows: m.query_tile_rows,
      query_metadata_version: m.query_metadata_version,
      heads_per_cta: m.heads_per_cta,
      decode_execution: m.decode_execution,
      prefill_exponential_law: m.prefill_exponential_law,
      module_digest: m.module_label,
      module_bytes: m.module_bytes,
      row_variant: m.row_variant,
    }),
  )
  runtime_attention_route_scope(model_content, model_plan, target, metadata)
}

///|
/// Binds measurements to the exact model plan, target and both compiled
/// attention implementations. Device assignment remains deployment-owned.
pub fn ApprovedReusableFusedRuntimeBundle::attention_route_scope(
  self : ApprovedReusableFusedRuntimeBundle,
) -> String raise FusedArtifactAdmissionError {
  runtime_attention_route_scope(
    self.model_content,
    self.model_plan,
    self.target,
    self.modules,
  )
}

///|
fn runtime_attention_route_scope(
  model_content : @luna_cuda_fused_parallel_aot.FusedParallelDigest,
  model_plan : @luna_cuda_fused_parallel_aot.FusedParallelDigest,
  target : @catalog.DeviceTarget,
  modules : ReadOnlyArray[ApprovedReusableFusedRuntimeModule],
) -> String raise FusedArtifactAdmissionError {
  guard modules
    .iter()
    .any(m => {
      m.abi
      is (QwenPagedAttentionReadOnlyProductionV2
      | QwenPagedAttentionReadOnlyDecodeSplitProductionV3)
    }) &&
    modules
    .iter()
    .any(m => m.abi == QwenPagedAttentionPrefillWideQueryProductionV5) else {
    raise InvalidFusedArtifact(Binding)
  }
  let out = StringBuilder()
  let scope_version = if modules.iter().any(m => m.row_variant is Some(_)) {
    "attention-chain-scope-v2\n"
  } else {
    "attention-chain-scope-v1\n"
  }
  out.write_string(
    scope_version + model_content.as_hex() + "\n" + model_plan.as_hex() + "\n",
  )
  reusable_runtime_append_target(out, target)
  for module_ in modules {
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
    if module_.heads_per_cta != 1 {
      out.write_string(
        "heads_per_cta=" + module_.heads_per_cta.to_string() + "\n",
      )
    }
    if module_.row_variant is Some(variant) {
      // Scope selection by declared code identity and execution geometry, not
      // by serializing and rehashing the already loaded CUBIN a second time.
      out.write_string(
        "row_variant_label=" + variant.module_label.as_hex() + "\n",
      )
      out.write_string("row_variant_symbol=" + variant.symbol + "\n")
      let launch = variant.launch
      for
        value in [
          variant.maximum_token_rows,
          variant.query_tile_rows,
          launch.grid_x(),
          launch.grid_y(),
          launch.grid_z(),
          launch.block_x(),
          launch.block_y(),
          launch.block_z(),
          launch.shared_memory_bytes(),
        ] {
        out.write_string(value.to_string() + ":")
      }
      out.write_string("\n")
    }
  }
  @crypto.bytes_to_hex_string(@crypto.sha256(@utf8.encode(out.to_string())))
}
