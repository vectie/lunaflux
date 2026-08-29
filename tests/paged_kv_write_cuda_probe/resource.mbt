///|
priv struct KvProbeBalance {
  mut contexts : Int
  mut streams : Int
  mut allocations : Int
  mut modules : Int
  mut functions : Int
  mut device_bytes : Int64
  mut pending : Int
  mut cleanup : Int
}

///|
fn kv_probe_balance() -> KvProbeBalance {
  {
    contexts: 0,
    streams: 0,
    allocations: 0,
    modules: 0,
    functions: 0,
    device_bytes: 0L,
    pending: 0,
    cleanup: 0,
  }
}

///|
fn KvProbeBalance::is_zero(self : KvProbeBalance) -> Bool {
  self.contexts == 0 &&
  self.streams == 0 &&
  self.allocations == 0 &&
  self.modules == 0 &&
  self.functions == 0 &&
  self.device_bytes == 0L &&
  self.pending == 0 &&
  self.cleanup == 0
}

///|
fn KvProbeBalance::render(self : KvProbeBalance) -> String {
  "context\{self.contexts},stream\{self.streams},allocation\{self.allocations},module\{self.modules},function\{self.functions},device_bytes\{self.device_bytes},pending\{self.pending},cleanup\{self.cleanup}"
}

///|
fn kv_launch(
  function_ : @device.Function,
  dimensions : @device.KernelLaunchDimensions,
  arguments : @device.KernelArguments,
  stream : @device.Stream,
  balance : KvProbeBalance,
) -> Unit raise {
  balance.pending += 1
  try function_.launch_synchronous(dimensions, arguments, stream) catch {
    error => {
      balance.pending -= 1
      raise error
    }
  } noraise {
    _ => balance.pending -= 1
  }
}
