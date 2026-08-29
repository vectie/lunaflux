///|
fn require_line(
  lines : Array[StringView],
  index : Int,
  key : StringView,
) -> StringView raise DeploymentBundleError {
  let line = lines[index]
  let prefix = "\{key}="
  if line.length() <= prefix.length() ||
    line[0:prefix.length()].to_owned() != prefix {
    raise NonCanonicalEvidence
  }
  line[prefix.length():]
}

///|
fn parse_architecture(
  value : StringView,
) -> DeploymentBundleArchitecture raise DeploymentBundleError {
  match value {
    "x86_64" => X86_64
    "aarch64" => Aarch64
    _ => raise InvalidField(Architecture)
  }
}

///|
fn parse_canonical_evidence(
  bytes : Bytes,
  digest : DeploymentBundleDigest,
  limits : DeploymentBundleLimits,
) -> DeploymentBundleEvidence raise DeploymentBundleError {
  if bytes.length() == 0 || bytes.length() > limits.maximum_bytes() {
    raise EvidenceTooLarge
  }
  let text = @utf8.decode(bytes) catch { _ => raise InvalidUtf8 }
  if text.contains("\r") || !text.has_suffix("\n") {
    raise NonCanonicalEvidence
  }
  let lines = text.split("\n").to_array()
  if lines.length() != 22 || lines[21].length() != 0 {
    raise NonCanonicalEvidence
  }
  if require_line(lines, 0, "schema") != "lunaflux.deployment-bundle.v1" {
    raise InvalidField(Evidence)
  }
  let base_image = require_line(lines, 1, "base_image")
  validate_base_image(base_image, limits)
  let architecture = parse_architecture(
    require_line(lines, 2, "linux_architecture"),
  )
  let launch_file_sha256 = require_digest(
    require_line(lines, 3, "launch_file_sha256"),
    LaunchFile,
  )
  let runtime_descriptor_relative = require_line(
    lines, 4, "runtime_descriptor_relative",
  )
  validate_relative_locator(
    runtime_descriptor_relative,
    RuntimeDescriptor,
    limits,
  )
  let runtime_descriptor_sha256 = require_digest(
    require_line(lines, 5, "runtime_descriptor_sha256"),
    RuntimeDescriptor,
  )
  let instance_policy_relative = require_line(
    lines, 6, "instance_policy_relative",
  )
  validate_relative_locator(instance_policy_relative, InstancePolicy, limits)
  let instance_policy_sha256 = require_digest(
    require_line(lines, 7, "instance_policy_sha256"),
    InstancePolicy,
  )
  let lunaflux_executable_sha256 = require_digest(
    require_line(lines, 8, "lunaflux_executable_sha256"),
    Executable,
  )
  let worker_executable_sha256 = require_digest(
    require_line(lines, 9, "worker_executable_sha256"),
    Executable,
  )
  let model_inventory_sha256 = require_digest(
    require_line(lines, 10, "model_inventory_sha256"),
    ModelInventory,
  )
  let policy_inventory_sha256 = require_digest(
    require_line(lines, 11, "policy_inventory_sha256"),
    PolicyInventory,
  )
  let kernel_manifest_relative = require_line(
    lines, 12, "kernel_manifest_relative",
  )
  validate_relative_locator(kernel_manifest_relative, KernelManifest, limits)
  if !kernel_manifest_relative.has_suffix(".json") {
    raise InvalidField(KernelManifest)
  }
  let kernel_manifest_sha256 = require_digest(
    require_line(lines, 13, "kernel_manifest_sha256"),
    KernelManifest,
  )
  let kernel_inventory_sha256 = require_digest(
    require_line(lines, 14, "kernel_inventory_sha256"),
    KernelInventory,
  )
  let runtime_library_inventory_sha256 = require_digest(
    require_line(lines, 15, "runtime_library_inventory_sha256"),
    RuntimeLibraryInventory,
  )
  let rootfs_inventory_sha256 = require_digest(
    require_line(lines, 16, "rootfs_inventory_sha256"),
    RootfsInventory,
  )
  let assembly_input_sha256 = require_digest(
    require_line(lines, 17, "assembly_input_sha256"),
    AssemblyInput,
  )
  let assembler_sha256 = require_digest(
    require_line(lines, 18, "assembler_sha256"),
    Assembler,
  )
  if require_line(lines, 19, "exact_inventory") != "1" ||
    require_line(lines, 20, "compiler_jit_free") != "1" {
    raise InvalidField(Policy)
  }
  {
    digest,
    base_image: base_image.to_owned(),
    architecture,
    launch_file_sha256,
    runtime_descriptor_relative: runtime_descriptor_relative.to_owned(),
    runtime_descriptor_sha256,
    instance_policy_relative: instance_policy_relative.to_owned(),
    instance_policy_sha256,
    lunaflux_executable_sha256,
    worker_executable_sha256,
    model_inventory_sha256,
    policy_inventory_sha256,
    kernel_manifest_relative: kernel_manifest_relative.to_owned(),
    kernel_manifest_sha256,
    kernel_inventory_sha256,
    runtime_library_inventory_sha256,
    rootfs_inventory_sha256,
    assembly_input_sha256,
    assembler_sha256,
    canonical: bytes,
  }
}
