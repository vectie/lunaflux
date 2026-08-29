///|
const ABSOLUTE_MAX_MODULES : Int = 4096

///|
const ABSOLUTE_MAX_ENTRY_POINTS : Int = 65536

///|
const ABSOLUTE_MAX_OWNED_BYTES : Int64 = 2147483647L

///|
const ABSOLUTE_MAX_SYMBOL_BYTES : Int = 1024

///|
pub struct KernelArtifactLimits {
  priv max_modules : Int
  priv max_entry_points : Int
  priv max_module_bytes : Int64
  priv max_total_module_bytes : Int64
  priv max_function_symbol_bytes : Int
}

///|
pub fn KernelArtifactLimits::new(
  max_modules~ : Int,
  max_entry_points~ : Int,
  max_module_bytes~ : Int64,
  max_total_module_bytes~ : Int64,
  max_function_symbol_bytes~ : Int,
) -> KernelArtifactLimits raise KernelArtifactError {
  if max_modules <= 0 || max_modules > ABSOLUTE_MAX_MODULES {
    raise InvalidLimit(ModuleCount)
  }
  if max_entry_points <= 0 || max_entry_points > ABSOLUTE_MAX_ENTRY_POINTS {
    raise InvalidLimit(EntryPointCount)
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
    max_modules,
    max_entry_points,
    max_module_bytes,
    max_total_module_bytes,
    max_function_symbol_bytes,
  }
}

///|
pub fn KernelArtifactLimits::max_modules(self : KernelArtifactLimits) -> Int {
  self.max_modules
}

///|
pub fn KernelArtifactLimits::max_entry_points(
  self : KernelArtifactLimits,
) -> Int {
  self.max_entry_points
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
