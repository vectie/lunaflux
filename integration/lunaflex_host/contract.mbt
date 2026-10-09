///|
/// LunaFlux's compile-time requirement for an optional linked LunaFlex core.
pub fn required_core_abi_version() -> Int {
  1
}

///|
/// Version of the LunaFlex module selected by this integration workspace.
pub fn linked_core_version() -> String {
  @lunaflex.core_version()
}

///|
/// Proves that this host is linked against LunaFlux's inference contracts as
/// well as LunaFlex. The returned scalar has no runtime authority.
pub fn linked_lunaflux_contract_probe() -> UInt64 raise {
  @inference.RequestId::new(1UL).value()
}

///|
/// Returns whether the selected LunaFlex module satisfies LunaFlux's initial
/// single-device dense-BF16 inference envelope.
pub fn is_linked_core_compatible() -> Bool {
  @lunaflex.is_core_compatible(required_core_abi_version(), [
    DenseBf16,
    SingleCudaDevice,
    ContinuousBatching,
    PagedKvMetadata,
    PrefixReuse,
    AotExecution,
  ])
}

///|
pub(all) suberror LunaflexCoreIntegrationError {
  IncompatibleCore(required_abi~ : Int, linked_version~ : String)
} derive(Debug, Eq)

///|
/// Startup/build acceptance gate for hosts that enable LunaFlex-backed
/// execution.
pub fn require_linked_core() -> Unit raise LunaflexCoreIntegrationError {
  if !is_linked_core_compatible() {
    raise IncompatibleCore(
      required_abi=required_core_abi_version(),
      linked_version=linked_core_version(),
    )
  }
}
