///|
const ABSOLUTE_MAX_ARTIFACTS : Int = 4096

///|
const ABSOLUTE_MAX_OWNED_BYTES : Int64 = 2147483647L

///|
const ABSOLUTE_MAX_SYMBOL_BYTES : Int = 1024

///|
/// Narrow allocation and identifier envelope for one admitted bundle.
pub struct KernelArtifactLimits {
  max_artifacts : Int
  max_module_bytes : Int64
  max_total_module_bytes : Int64
  max_function_symbol_bytes : Int
} derive(Debug, Eq)

///|
/// Constructs hard-capped limits that remain exactly representable by native
/// MoonBit `Bytes` lengths and the downstream checked ABI.
pub fn KernelArtifactLimits::new(
  max_artifacts~ : Int,
  max_module_bytes~ : Int64,
  max_total_module_bytes~ : Int64,
  max_function_symbol_bytes~ : Int,
) -> KernelArtifactLimits raise KernelArtifactError {
  if max_artifacts <= 0 || max_artifacts > ABSOLUTE_MAX_ARTIFACTS {
    raise InvalidLimit(ArtifactCount)
  }
  if max_module_bytes <= 0L || max_module_bytes > ABSOLUTE_MAX_OWNED_BYTES {
    raise InvalidLimit(ModuleBytes)
  }
  if max_total_module_bytes <= 0L ||
    max_total_module_bytes > ABSOLUTE_MAX_OWNED_BYTES {
    raise InvalidLimit(TotalModuleBytes)
  }
  if max_function_symbol_bytes <= 0 ||
    max_function_symbol_bytes > ABSOLUTE_MAX_SYMBOL_BYTES {
    raise InvalidLimit(FunctionSymbolBytes)
  }
  {
    max_artifacts,
    max_module_bytes,
    max_total_module_bytes,
    max_function_symbol_bytes,
  }
}

///|
pub fn KernelArtifactLimits::max_artifacts(self : KernelArtifactLimits) -> Int {
  self.max_artifacts
}

///|
pub fn KernelArtifactLimits::max_module_bytes(
  self : KernelArtifactLimits,
) -> Int64 {
  self.max_module_bytes
}

///|
pub fn KernelArtifactLimits::max_total_module_bytes(
  self : KernelArtifactLimits,
) -> Int64 {
  self.max_total_module_bytes
}

///|
pub fn KernelArtifactLimits::max_function_symbol_bytes(
  self : KernelArtifactLimits,
) -> Int {
  self.max_function_symbol_bytes
}
