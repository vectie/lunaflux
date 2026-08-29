///|
struct Fp8V2ProbeBalance {
  mut contexts : Int
  mut streams : Int
  mut executors : Int
  mut allocations : Int
  mut modules : Int
  mut functions : Int
  mut device_bytes : Int64
  mut pending : Int
  mut cleanup : Int
}

///|
fn fp8_probe_balance() -> Fp8V2ProbeBalance {
  {
    contexts: 0,
    streams: 0,
    executors: 0,
    allocations: 0,
    modules: 0,
    functions: 0,
    device_bytes: 0L,
    pending: 0,
    cleanup: 0,
  }
}

///|
fn Fp8V2ProbeBalance::is_zero(self : Fp8V2ProbeBalance) -> Bool {
  self.contexts == 0 &&
  self.streams == 0 &&
  self.executors == 0 &&
  self.allocations == 0 &&
  self.modules == 0 &&
  self.functions == 0 &&
  self.device_bytes == 0L &&
  self.pending == 0 &&
  self.cleanup == 0
}
