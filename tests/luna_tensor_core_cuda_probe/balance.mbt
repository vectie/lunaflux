///|
struct LunaTensorCoreProbeBalance {
  mut contexts : Int
  mut streams : Int
  mut allocations : Int
  mut modules : Int
  mut functions : Int
  mut device_bytes : Int64
  mut cleanup_failures : Int
}

///|
fn luna_tensor_core_probe_balance() -> LunaTensorCoreProbeBalance {
  {
    contexts: 0,
    streams: 0,
    allocations: 0,
    modules: 0,
    functions: 0,
    device_bytes: 0L,
    cleanup_failures: 0,
  }
}

///|
fn LunaTensorCoreProbeBalance::is_zero(
  self : LunaTensorCoreProbeBalance,
) -> Bool {
  self.contexts == 0 &&
  self.streams == 0 &&
  self.allocations == 0 &&
  self.modules == 0 &&
  self.functions == 0 &&
  self.device_bytes == 0L &&
  self.cleanup_failures == 0
}
