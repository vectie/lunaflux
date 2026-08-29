///|
priv struct FusedDeploymentReader {
  lines : Array[String]
  mut cursor : Int
}

///|
fn fused_deployment_reader(
  bytes : Bytes,
) -> FusedDeploymentReader raise FusedArtifactAdmissionError {
  if bytes.is_empty() || bytes.length() > 33554432 {
    raise InvalidFusedArtifact(Manifest)
  }
  let text = @utf8.decode(bytes) catch {
    _ => raise InvalidFusedArtifact(Manifest)
  }
  let lines = text.split("\n").map(line => line.to_owned()).to_array()
  if lines.length() < 3 || lines[lines.length() - 1] != "" {
    raise InvalidFusedArtifact(Manifest)
  }
  { lines, cursor: 0 }
}

///|
fn FusedDeploymentReader::value(
  self : FusedDeploymentReader,
  key : String,
) -> String raise FusedArtifactAdmissionError {
  if self.cursor >= self.lines.length() - 1 {
    raise InvalidFusedArtifact(Manifest)
  }
  let prefix = key + "="
  let line = self.lines[self.cursor]
  self.cursor += 1
  if !line.has_prefix(prefix) || line.length() == prefix.length() {
    raise InvalidFusedArtifact(Manifest)
  }
  line[prefix.length():].to_owned()
}

///|
fn FusedDeploymentReader::int(
  self : FusedDeploymentReader,
  key : String,
) -> Int raise FusedArtifactAdmissionError {
  let text = self.value(key)
  let value : Int = @strconv.from_str(text) catch {
    _ => raise InvalidFusedArtifact(Manifest)
  }
  if value.to_string() != text {
    raise InvalidFusedArtifact(Manifest)
  }
  value
}

///|
fn FusedDeploymentReader::uint(
  self : FusedDeploymentReader,
  key : String,
) -> UInt raise FusedArtifactAdmissionError {
  let text = self.value(key)
  let value : UInt = @strconv.from_str(text) catch {
    _ => raise InvalidFusedArtifact(Manifest)
  }
  if value.to_string() != text {
    raise InvalidFusedArtifact(Manifest)
  }
  value
}

///|
fn fused_deployment_digest(
  text : String,
) -> @luna_cuda_fused_parallel_aot.FusedParallelDigest raise FusedArtifactAdmissionError {
  @luna_cuda_fused_parallel_aot.FusedParallelDigest::from_sha256(text) catch {
    _ => raise InvalidFusedArtifact(Manifest)
  }
}

///|
fn fused_deployment_hex_nibble(
  unit : UInt16,
) -> Byte raise FusedArtifactAdmissionError {
  if unit >= 48 && unit <= 57 {
    (unit.to_int() - '0'.to_int()).to_byte()
  } else if unit >= 97 && unit <= 102 {
    (unit.to_int() - 'a'.to_int() + 10).to_byte()
  } else {
    raise InvalidFusedArtifact(Module)
  }
}

///|
fn fused_deployment_decode_hex(
  text : String,
) -> Bytes raise FusedArtifactAdmissionError {
  if text.is_empty() || text.length() % 2 != 0 {
    raise InvalidFusedArtifact(Module)
  }
  Bytes::makei(text.length() / 2, index => {
    (fused_deployment_hex_nibble(text[index * 2]) << 4) |
    fused_deployment_hex_nibble(text[index * 2 + 1])
  })
}

///|
fn fused_deployment_append_fallbacks(
  out : StringBuilder,
  prefix : String,
  fallbacks : ReadOnlyArray[FusedStandaloneFallbackIdentity],
) -> Unit {
  out.write_string(
    prefix + "fallback_count=" + fallbacks.length().to_string() + "\n",
  )
  for index, fallback in fallbacks {
    let item = prefix + "fallback_" + index.to_string() + "_"
    let role = match fallback.role() {
      QkvProjectionFallback => "QkvProjectionFallback"
      PositionedRopeFallback => "PositionedRopeFallback"
      PagedKvWriteFallback => "PagedKvWriteFallback"
      ResidualAddFallback => "ResidualAddFallback"
      RmsNormFallback => "RmsNormFallback"
    }
    out.write_string(item + "role=" + role + "\n")
    out.write_string(
      item + "operation=" + fallback.operation_index().to_string() + "\n",
    )
    out.write_string(
      item + "source_sha256=" + fallback.source_digest().as_hex() + "\n",
    )
    out.write_string(
      item + "recipe_sha256=" + fallback.recipe_digest().as_hex() + "\n",
    )
  }
}

///|
fn fused_deployment_append_module(
  out : StringBuilder,
  prefix : String,
  module_ : ApprovedFusedKernelModule,
) -> Unit raise FusedArtifactAdmissionError {
  if module_.function_symbol().contains("\n") {
    raise InvalidFusedArtifact(Symbol)
  }
  let launch = module_.launch_dimensions()
  out.write_string(
    prefix + "candidate_sha256=" + module_.candidate_digest().as_hex() + "\n",
  )
  out.write_string(
    prefix + "source_sha256=" + module_.source_digest().as_hex() + "\n",
  )
  out.write_string(
    prefix + "recipe_sha256=" + module_.recipe_digest().as_hex() + "\n",
  )
  out.write_string(
    prefix + "toolchain_sha256=" + module_.toolchain_digest_sha256() + "\n",
  )
  out.write_string(
    prefix + "driver_sha256=" + module_.driver_identity_sha256() + "\n",
  )
  out.write_string(prefix + "symbol=" + module_.function_symbol() + "\n")
  out.write_string(
    prefix + "abi=" + module_.kernel_abi().canonical_name() + "\n",
  )
  out.write_string(
    prefix +
    "purpose=" +
    fused_artifact_purpose_name(module_.artifact_purpose()) +
    "\n",
  )
  out.write_string(
    prefix + "canary=" + module_.dispatch_canary_per_token().to_string() + "\n",
  )
  out.write_string(prefix + "grid_x=" + launch.grid_x().to_string() + "\n")
  out.write_string(prefix + "grid_y=" + launch.grid_y().to_string() + "\n")
  out.write_string(prefix + "grid_z=" + launch.grid_z().to_string() + "\n")
  out.write_string(prefix + "block_x=" + launch.block_x().to_string() + "\n")
  out.write_string(prefix + "block_y=" + launch.block_y().to_string() + "\n")
  out.write_string(prefix + "block_z=" + launch.block_z().to_string() + "\n")
  out.write_string(
    prefix + "shared_bytes=" + launch.shared_memory_bytes().to_string() + "\n",
  )
  out.write_string(
    prefix + "module_sha256=" + module_.module_digest().as_hex() + "\n",
  )
  out.write_string(
    prefix +
    "receipt_sha256=" +
    module_.compile_receipt_digest().as_hex() +
    "\n",
  )
  out.write_string(
    prefix +
    "binding_sha256=" +
    module_.compiled_binding_digest().as_hex() +
    "\n",
  )
  out.write_string(
    prefix + "benchmark_sha256=" + module_.benchmark_digest().as_hex() + "\n",
  )
  out.write_string(
    prefix +
    "module_hex=" +
    @crypto.bytes_to_hex_string(module_.module_bytes()) +
    "\n",
  )
  fused_deployment_append_fallbacks(out, prefix, module_.fallbacks())
}

///|
fn fused_deployment_canonical(
  catalog : ApprovedFusedKernelCatalog,
  bundle : ApprovedFusedKernelBundle,
) -> Bytes raise FusedArtifactAdmissionError {
  if catalog.manifest_digest() != bundle.manifest_digest() ||
    catalog.target() != bundle.target() ||
    catalog.qkv.module_digest() != bundle.qkv.module_digest() ||
    catalog.qkv.compiled_binding_digest() !=
    bundle.qkv.compiled_binding_digest() ||
    catalog.residual.module_digest() != bundle.residual.module_digest() ||
    catalog.residual.compiled_binding_digest() !=
    bundle.residual.compiled_binding_digest() {
    raise MismatchedFusedArtifact(Catalog)
  }
  let target = catalog.target()
  let out = StringBuilder()
  out.write_string("schema=lunaflux-fused-runtime-source.v1\n")
  out.write_string(
    "manifest_sha256=" + catalog.manifest_digest().as_hex() + "\n",
  )
  out.write_string("catalog_sha256=" + catalog.digest().as_hex() + "\n")
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
  fused_deployment_append_module(out, "qkv_", bundle.qkv_module())
  fused_deployment_append_module(out, "residual_", bundle.residual_module())
  @utf8.encode(out.to_string())
}

///|
/// Offline export from already approved opaque identities. It performs no file
/// write and grants no deployment or device authority.
pub fn export_digest_pinned_fused_runtime_source(
  catalog : ApprovedFusedKernelCatalog,
  bundle : ApprovedFusedKernelBundle,
) -> Bytes raise FusedArtifactAdmissionError {
  fused_deployment_canonical(catalog, bundle)
}

///|
fn fused_deployment_read_fallbacks(
  reader : FusedDeploymentReader,
  prefix : String,
) -> ReadOnlyArray[FusedStandaloneFallbackIdentity] raise FusedArtifactAdmissionError {
  let count = reader.int(prefix + "fallback_count")
  if count < 0 || count > 8 {
    raise InvalidFusedArtifact(Fallback)
  }
  let values : Array[FusedStandaloneFallbackIdentity] = []
  for index in 0..<count {
    let item = prefix + "fallback_" + index.to_string() + "_"
    let role_text = reader.value(item + "role")
    let role = match role_text {
      "QkvProjectionFallback" => QkvProjectionFallback
      "PositionedRopeFallback" => PositionedRopeFallback
      "PagedKvWriteFallback" => PagedKvWriteFallback
      "ResidualAddFallback" => ResidualAddFallback
      "RmsNormFallback" => RmsNormFallback
      _ => raise InvalidFusedArtifact(Fallback)
    }
    values.push(
      FusedStandaloneFallbackIdentity::new(
        role~,
        operation=reader.int(item + "operation"),
        source=fused_deployment_digest(reader.value(item + "source_sha256")),
        recipe=fused_deployment_digest(reader.value(item + "recipe_sha256")),
      ),
    )
  }
  ReadOnlyArray::from_array(values)
}

///|
fn fused_deployment_read_module(
  reader : FusedDeploymentReader,
  prefix : String,
  family : @luna_cuda_fused_parallel_aot.FusedParallelFamily,
  target : @catalog.DeviceTarget,
) -> ApprovedFusedKernelModule raise FusedArtifactAdmissionError {
  let candidate = fused_deployment_digest(
    reader.value(prefix + "candidate_sha256"),
  )
  let source = fused_deployment_digest(reader.value(prefix + "source_sha256"))
  let recipe = fused_deployment_digest(reader.value(prefix + "recipe_sha256"))
  let toolchain = reader.value(prefix + "toolchain_sha256")
  let driver = reader.value(prefix + "driver_sha256")
  let symbol = reader.value(prefix + "symbol")
  let abi_text = reader.value(prefix + "abi")
  let purpose_text = reader.value(prefix + "purpose")
  let purpose = match purpose_text {
    "production-fast-path" => @luna_cuda_fused_parallel_aot.ProductionFastPath
    "qualification-diagnostics" => QualificationDiagnostics
    _ => raise InvalidFusedArtifact(KernelAbi)
  }
  let abi = match abi_text {
    "qkv-positioned-rope-paged-kv-write-production-raw-pointer-v2" =>
      @luna_cuda_fused_parallel_aot.QkvPositionedRopePagedKvWriteProductionAbiV2
    "qkv-positioned-rope-paged-kv-write-raw-pointer-v1" =>
      QkvPositionedRopePagedKvWriteAbiV1
    "residual-rmsnorm-production-raw-pointer-v2" =>
      ResidualRmsNormProductionAbiV2
    "residual-rmsnorm-raw-pointer-v1" => ResidualRmsNormAbiV1
    _ => raise InvalidFusedArtifact(KernelAbi)
  }
  let canary = reader.uint(prefix + "canary")
  let launch = @launch_contract.AotLaunchDimensions::new(
    grid_x=reader.int(prefix + "grid_x"),
    grid_y=reader.int(prefix + "grid_y"),
    grid_z=reader.int(prefix + "grid_z"),
    block_x=reader.int(prefix + "block_x"),
    block_y=reader.int(prefix + "block_y"),
    block_z=reader.int(prefix + "block_z"),
    shared_memory_bytes=reader.int(prefix + "shared_bytes"),
  ) catch {
    _ => raise InvalidFusedArtifact(Launch)
  }
  let module_digest = fused_deployment_digest(
    reader.value(prefix + "module_sha256"),
  )
  let receipt = fused_deployment_digest(reader.value(prefix + "receipt_sha256"))
  let binding = fused_deployment_digest(reader.value(prefix + "binding_sha256"))
  let benchmark = fused_deployment_digest(
    reader.value(prefix + "benchmark_sha256"),
  )
  let artifact_bytes = fused_deployment_decode_hex(
    reader.value(prefix + "module_hex"),
  )
  let fallbacks = fused_deployment_read_fallbacks(reader, prefix)
  if toolchain.length() != 64 ||
    driver.length() != 64 ||
    symbol.contains("\n") ||
    fused_artifact_digest(artifact_bytes) != module_digest ||
    abi != fused_expected_abi(family, purpose) {
    raise MismatchedFusedArtifact(Module)
  }
  fused_validate_fallbacks(family, fallbacks)
  match purpose {
    QualificationDiagnostics =>
      if canary == 0U {
        raise MismatchedFusedArtifact(KernelAbi)
      }
    ProductionFastPath =>
      if canary != 0U {
        raise MismatchedFusedArtifact(KernelAbi)
      }
  }
  {
    family,
    candidate,
    source,
    recipe,
    toolchain,
    driver,
    target,
    launch,
    symbol,
    abi,
    purpose,
    dispatch_canary_per_token: canary,
    artifact_bytes,
    module_digest,
    receipt_digest: receipt,
    binding_digest: binding,
    benchmark_digest: benchmark,
    fallbacks,
  }
}

///|
/// Startup-only re-admission of an exact descriptor snapshot already pinned by
/// the deployment/bootstrap source. No path, filesystem, or live device input
/// is accepted here.
pub fn admit_digest_pinned_fused_runtime_source(
  canonical : Bytes,
  expected_sha256 : String,
) -> (ApprovedFusedKernelCatalog, ApprovedFusedKernelBundle) raise FusedArtifactAdmissionError {
  if @crypto.bytes_to_hex_string(@crypto.sha256(canonical)) != expected_sha256 {
    raise MismatchedFusedArtifact(Manifest)
  }
  let reader = fused_deployment_reader(canonical)
  if reader.value("schema") != "lunaflux-fused-runtime-source.v1" {
    raise InvalidFusedArtifact(Manifest)
  }
  let manifest = fused_deployment_digest(reader.value("manifest_sha256"))
  let expected_catalog = fused_deployment_digest(reader.value("catalog_sha256"))
  let target = @catalog.DeviceTarget::new(
    compute_major=reader.int("compute_major"),
    compute_minor=reader.int("compute_minor"),
    supports_bf16=reader.int("supports_bf16") == 1,
    supports_cublas_lt=reader.int("supports_cublas_lt") == 1,
  ) catch {
    _ => raise InvalidFusedArtifact(Target)
  }
  let qkv = fused_deployment_read_module(
    reader,
    "qkv_",
    QkvPositionedRopePagedKvWrite,
    target,
  )
  let residual = fused_deployment_read_module(
    reader,
    "residual_",
    ResidualRmsNorm,
    target,
  )
  if reader.cursor != reader.lines.length() - 1 ||
    qkv.toolchain != residual.toolchain ||
    qkv.driver != residual.driver {
    raise MismatchedFusedArtifact(Compiler)
  }
  let bundle : ApprovedFusedKernelBundle = { manifest, target, qkv, residual }
  let catalog_canonical = fused_catalog_canonical(
    manifest, target, qkv, residual,
  )
  let catalog : ApprovedFusedKernelCatalog = {
    manifest,
    target,
    qkv,
    residual,
    canonical: catalog_canonical,
    digest: fused_artifact_digest(catalog_canonical),
  }
  if catalog.digest() != expected_catalog ||
    fused_deployment_canonical(catalog, bundle) != canonical {
    raise MismatchedFusedArtifact(Catalog)
  }
  (catalog, bundle)
}
