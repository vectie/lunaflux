///|
/// Re-admits a compact production sidecar containing one deterministic CUBIN.
/// The enclosing bootstrap already pins these exact canonical bytes.
pub fn admit_digest_pinned_reusable_fused_residual_runtime(
  canonical : Bytes,
  expected_sha256 : String,
) -> ApprovedReusableFusedResidualRuntime raise FusedArtifactAdmissionError {
  if @crypto.bytes_to_hex_string(@crypto.sha256(canonical)) != expected_sha256 {
    raise MismatchedFusedArtifact(Manifest)
  }
  let reader = fused_deployment_reader(canonical)
  if reader.value("schema") !=
    "lunaflux-reusable-fused-residual-rmsnorm-runtime.v1" {
    raise InvalidFusedArtifact(Manifest)
  }
  let model_content = fused_deployment_digest(
    reader.value("model_content_sha256"),
  )
  let model_plan = fused_deployment_digest(reader.value("model_plan_sha256"))
  let target = @catalog.DeviceTarget::new(
    compute_major=reader.int("compute_major"),
    compute_minor=reader.int("compute_minor"),
    supports_bf16=reader.int("supports_bf16") == 1,
    supports_cublas_lt=reader.int("supports_cublas_lt") == 1,
  ) catch {
    _ => raise InvalidFusedArtifact(Target)
  }
  let residual_operation = reader.int("residual_operation")
  let norm_operation = reader.int("norm_operation")
  let symbol = reader.value("function_symbol")
  let launch = @launch_contract.AotLaunchDimensions::new(
    grid_x=reader.int("grid_x"),
    grid_y=reader.int("grid_y"),
    grid_z=reader.int("grid_z"),
    block_x=reader.int("block_x"),
    block_y=reader.int("block_y"),
    block_z=reader.int("block_z"),
    shared_memory_bytes=reader.int("shared_bytes"),
  ) catch {
    _ => raise InvalidFusedArtifact(Launch)
  }
  let module_digest = fused_deployment_digest(reader.value("module_sha256"))
  let module_bytes = fused_deployment_decode_hex(reader.value("module_hex"))
  if reader.cursor != reader.lines.length() - 1 ||
    residual_operation < 0 ||
    norm_operation != residual_operation + 1 ||
    symbol.is_empty() ||
    symbol.contains("\n") ||
    launch.block_x() != 128 ||
    launch.block_y() != 1 ||
    launch.block_z() != 1 ||
    launch.grid_y() != 1 ||
    launch.grid_z() != 1 ||
    launch.shared_memory_bytes() != 512 ||
    fused_artifact_digest(module_bytes) != module_digest {
    raise MismatchedFusedArtifact(Module)
  }
  {
    source_digest: fused_deployment_digest(expected_sha256),
    model_content,
    model_plan,
    target,
    residual_operation,
    norm_operation,
    symbol,
    launch,
    module_bytes,
  }
}

///|
/// Canonically packages one deterministic production CUBIN for startup-time
/// reuse across identical residual/RMSNorm shapes.
pub fn export_reusable_fused_residual_runtime(
  model_content_sha256~ : String,
  model_plan_sha256~ : String,
  target~ : @catalog.DeviceTarget,
  residual_operation~ : Int,
  norm_operation~ : Int,
  function_symbol~ : String,
  launch~ : @launch_contract.AotLaunchDimensions,
  module_bytes~ : Bytes,
) -> Bytes raise FusedArtifactAdmissionError {
  ignore(fused_deployment_digest(model_content_sha256))
  ignore(fused_deployment_digest(model_plan_sha256))
  if residual_operation < 0 ||
    norm_operation != residual_operation + 1 ||
    function_symbol.is_empty() ||
    function_symbol.contains("\n") ||
    module_bytes.is_empty() {
    raise InvalidFusedArtifact(Module)
  }
  let out = StringBuilder()
  out.write_string(
    "schema=lunaflux-reusable-fused-residual-rmsnorm-runtime.v1\n",
  )
  out.write_string("model_content_sha256=" + model_content_sha256 + "\n")
  out.write_string("model_plan_sha256=" + model_plan_sha256 + "\n")
  out.write_string("compute_major=" + target.compute_major().to_string() + "\n")
  out.write_string("compute_minor=" + target.compute_minor().to_string() + "\n")
  out.write_string(
    "supports_bf16=" + (if target.supports_bf16() { "1" } else { "0" }) + "\n",
  )
  out.write_string(
    "supports_cublas_lt=" +
    (if target.supports_cublas_lt() { "1" } else { "0" }) +
    "\n",
  )
  out.write_string(
    "residual_operation=" + residual_operation.to_string() + "\n",
  )
  out.write_string("norm_operation=" + norm_operation.to_string() + "\n")
  out.write_string("function_symbol=" + function_symbol + "\n")
  out.write_string("grid_x=" + launch.grid_x().to_string() + "\n")
  out.write_string("grid_y=" + launch.grid_y().to_string() + "\n")
  out.write_string("grid_z=" + launch.grid_z().to_string() + "\n")
  out.write_string("block_x=" + launch.block_x().to_string() + "\n")
  out.write_string("block_y=" + launch.block_y().to_string() + "\n")
  out.write_string("block_z=" + launch.block_z().to_string() + "\n")
  out.write_string(
    "shared_bytes=" + launch.shared_memory_bytes().to_string() + "\n",
  )
  out.write_string(
    "module_sha256=" + fused_artifact_digest(module_bytes).as_hex() + "\n",
  )
  out.write_string(
    "module_hex=" + @crypto.bytes_to_hex_string(module_bytes) + "\n",
  )
  let canonical = @utf8.encode(out.to_string())
  ignore(
    admit_digest_pinned_reusable_fused_residual_runtime(
      canonical,
      @crypto.bytes_to_hex_string(@crypto.sha256(canonical)),
    ),
  )
  canonical
}
