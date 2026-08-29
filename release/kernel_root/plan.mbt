///|
fn root_sha256(bytes : Bytes) -> String {
  @crypto.bytes_to_hex_string(@crypto.sha256(bytes))
}

///|
fn append_inventory_line(
  output : StringBuilder,
  digest : String,
  relative_path : String,
) -> Unit {
  output.write_string(digest)
  output.write_string("  ")
  output.write_string(relative_path)
  output.write_char('\n')
}

///|
/// Projects an already authenticated complete BF16 bundle into the sole file
/// set allowed under a deployment kernel root. This function performs no I/O.
pub fn plan_luna_bf16_kernel_root(
  release : @luna_bf16_kernel_producer.LunaBf16KernelReleaseInput,
  limits : LunaKernelRootLimits,
) -> LunaKernelRootPlan raise LunaKernelRootError {
  let bundle = release.bundle()
  let manifest = bundle.execution_manifest()
  if manifest.operation_count() != bundle.operations().length() ||
    manifest.module_count() != bundle.modules().length() ||
    bundle.modules().length() + 1 > limits.max_files {
    raise Invalid(Manifest)
  }
  let manifest_relative = "lunaflux.execution.json"
  let manifest_bytes = manifest.bytes()
  let manifest_sha256 = root_sha256(manifest_bytes)
  if manifest_sha256 != manifest.digest_sha256() {
    raise Invalid(Manifest)
  }
  let files : Array[LunaKernelRootFile] = [
    {
      relative_path: manifest_relative,
      digest_sha256: manifest_sha256,
      bytes: manifest_bytes,
    },
  ]
  let inventory = StringBuilder()
  append_inventory_line(inventory, manifest_sha256, manifest_relative)
  let mut previous = ""
  for module_item in bundle.modules() {
    let digest = module_item.digest().sha256()
    let expected = "sha256/" + digest + ".cubin"
    let bytes = module_item.module_bytes()
    if module_item.relative_path() != expected ||
      root_sha256(bytes) != digest ||
      (previous != "" && previous.compare(expected) >= 0) {
      raise Invalid(ModuleOrdering)
    }
    previous = expected
    files.push({ relative_path: expected, digest_sha256: digest, bytes })
    append_inventory_line(inventory, digest, expected)
  }
  let inventory_bytes = @utf8.encode(inventory.to_string())
  if inventory_bytes.length() > limits.max_inventory_bytes {
    raise LimitExceeded(
      Inventory,
      maximum=limits.max_inventory_bytes.to_int64(),
    )
  }
  let inventory_sha256 = root_sha256(inventory_bytes)
  let identity_material = @utf8.encode(
    "schema=lunaflux-bf16-kernel-root-identity.v1\n" +
    "execution_manifest_sha256=" +
    manifest_sha256 +
    "\nfiles_inventory_sha256=" +
    inventory_sha256 +
    "\n",
  )
  let identity_sha256 = root_sha256(identity_material)
  let evidence = @utf8.encode(
    "schema=lunaflux-bf16-kernel-root-plan.v1\n" +
    "execution_manifest_relative=" +
    manifest_relative +
    "\nexecution_manifest_sha256=" +
    manifest_sha256 +
    "\nmanifest_schema_version=2\n" +
    "catalog_version=3\n" +
    "operation_count=" +
    manifest.operation_count().to_string() +
    "\nmodule_count=" +
    manifest.module_count().to_string() +
    "\nentry_point_count=" +
    manifest.entry_point_count().to_string() +
    "\ntotal_module_bytes=" +
    bundle.total_module_bytes().to_string() +
    "\nfiles_inventory_sha256=" +
    inventory_sha256 +
    "\nkernel_root_identity_sha256=" +
    identity_sha256 +
    "\nexact_inventory=1\ncompiler_jit_free=1\n",
  )
  if evidence.length() > limits.max_evidence_bytes {
    raise LimitExceeded(Evidence, maximum=limits.max_evidence_bytes.to_int64())
  }
  {
    manifest_relative,
    manifest_sha256,
    files: ReadOnlyArray::from_array(files),
    inventory: inventory_bytes,
    inventory_sha256,
    evidence,
    evidence_sha256: root_sha256(evidence),
    identity_sha256,
  }
}
