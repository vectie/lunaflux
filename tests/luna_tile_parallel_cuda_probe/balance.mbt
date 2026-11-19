///|
struct LunaTileParallelProbeBalance {
  mut contexts : Int32
  mut streams : Int32
  mut allocations : Int32
  mut modules : Int32
  mut functions : Int32
  mut device_bytes : Int64
  mut cleanup_failures : Int32
}

///|
fn luna_tile_parallel_probe_balance() -> LunaTileParallelProbeBalance {
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
fn LunaTileParallelProbeBalance::is_zero(
  self : LunaTileParallelProbeBalance,
) -> Bool {
  self.contexts == 0 &&
  self.streams == 0 &&
  self.allocations == 0 &&
  self.modules == 0 &&
  self.functions == 0 &&
  self.device_bytes == 0L &&
  self.cleanup_failures == 0
}
