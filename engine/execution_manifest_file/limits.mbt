///|
const ABSOLUTE_MAX_MANIFEST_BYTES : Int64 = 67108864L

///|
const ABSOLUTE_MAX_JSON_DEPTH : Int = 64

///|
const ABSOLUTE_MAX_MODULES : Int = 4096

///|
const ABSOLUTE_MAX_OPERATIONS : Int = 4096

///|
/// Exact startup ceilings for manifest parsing and every derived admission.
pub struct ExecutionManifestFileLimits {
  priv max_manifest_bytes : Int64
  priv max_json_depth : Int
  priv max_modules : Int
  priv max_operations : Int
  priv device_memory : @device_memory.DeviceMemoryLimits
  priv launch_contract : @launch_contract.LaunchContractLimits
  priv artifact_file : @artifact_file.KernelArtifactFileLimits
}

///|
pub fn ExecutionManifestFileLimits::new(
  max_manifest_bytes~ : Int64,
  max_json_depth~ : Int,
  max_modules~ : Int,
  max_operations~ : Int,
  device_memory~ : @device_memory.DeviceMemoryLimits,
  launch_contract~ : @launch_contract.LaunchContractLimits,
  artifact_file~ : @artifact_file.KernelArtifactFileLimits,
) -> ExecutionManifestFileLimits raise ExecutionManifestFileError {
  if max_manifest_bytes <= 0L ||
    max_manifest_bytes > ABSOLUTE_MAX_MANIFEST_BYTES {
    raise InvalidLimit(ManifestBytes)
  }
  if max_json_depth <= 0 || max_json_depth > ABSOLUTE_MAX_JSON_DEPTH {
    raise InvalidLimit(JsonDepth)
  }
  if max_modules <= 0 || max_modules > ABSOLUTE_MAX_MODULES {
    raise InvalidLimit(ModuleCount)
  }
  if max_operations <= 0 || max_operations > ABSOLUTE_MAX_OPERATIONS {
    raise InvalidLimit(OperationCount)
  }
  let admitted_memory = @device_memory.DeviceMemoryLimits::new(
    activation_alignment=device_memory.activation_alignment(),
    max_arena_bytes=device_memory.max_arena_bytes(),
  ) catch {
    _ => raise InvalidLimit(DeviceMemory)
  }
  let admitted_contract = @launch_contract.LaunchContractLimits::new(
    max_profiles=launch_contract.max_profiles(),
    max_contracts=launch_contract.max_contracts(),
    max_operands_per_contract=launch_contract.max_operands_per_contract(),
  ) catch {
    _ => raise InvalidLimit(LaunchContract)
  }
  let artifact = artifact_file.artifact()
  let admitted_artifact = @artifact.KernelArtifactLimits::new(
    max_modules=artifact.max_modules(),
    max_entry_points=artifact.max_entry_points(),
    max_module_bytes=artifact.max_module_bytes(),
    max_total_module_bytes=artifact.max_total_module_bytes(),
    max_function_symbol_bytes=artifact.max_function_symbol_bytes(),
  ) catch {
    _ => raise InvalidLimit(ArtifactFile)
  }
  let admitted_artifact_file = @artifact_file.KernelArtifactFileLimits::new(
    max_manifest_bytes=artifact_file.max_manifest_bytes(),
    max_json_depth=artifact_file.max_json_depth(),
    max_path_bytes=artifact_file.max_path_bytes(),
    artifact=admitted_artifact,
  ) catch {
    _ => raise InvalidLimit(ArtifactFile)
  }
  if max_modules > admitted_artifact.max_modules() {
    raise InvalidLimit(ModuleCount)
  }
  if max_operations > admitted_contract.max_contracts() {
    raise InvalidLimit(OperationCount)
  }
  {
    max_manifest_bytes,
    max_json_depth,
    max_modules,
    max_operations,
    device_memory: admitted_memory,
    launch_contract: admitted_contract,
    artifact_file: admitted_artifact_file,
  }
}

///|
pub fn ExecutionManifestFileLimits::max_manifest_bytes(
  self : ExecutionManifestFileLimits,
) -> Int64 {
  self.max_manifest_bytes
}

///|
pub fn ExecutionManifestFileLimits::max_json_depth(
  self : ExecutionManifestFileLimits,
) -> Int {
  self.max_json_depth
}

///|
pub fn ExecutionManifestFileLimits::max_modules(
  self : ExecutionManifestFileLimits,
) -> Int {
  self.max_modules
}

///|
pub fn ExecutionManifestFileLimits::max_operations(
  self : ExecutionManifestFileLimits,
) -> Int {
  self.max_operations
}

///|
pub fn ExecutionManifestFileLimits::device_memory(
  self : ExecutionManifestFileLimits,
) -> @device_memory.DeviceMemoryLimits {
  self.device_memory
}

///|
pub fn ExecutionManifestFileLimits::launch_contract(
  self : ExecutionManifestFileLimits,
) -> @launch_contract.LaunchContractLimits {
  self.launch_contract
}

///|
pub fn ExecutionManifestFileLimits::artifact_file(
  self : ExecutionManifestFileLimits,
) -> @artifact_file.KernelArtifactFileLimits {
  self.artifact_file
}
