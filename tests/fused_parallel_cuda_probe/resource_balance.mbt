///|
priv struct FusedProbeResourceBalance {
  mut contexts : Int32
  mut streams : Int32
  events : Int32
  graphs : Int32
  graph_execs : Int32
  mut allocations : Int32
  mut modules : Int32
  mut functions : Int32
  mut executors : Int32
  mut device_bytes : Int64
  mut pending_work : Int32
  mut cleanup_failures : Int32
}

///|
fn fused_probe_resource_balance() -> FusedProbeResourceBalance {
  {
    contexts: 0,
    streams: 0,
    events: 0,
    graphs: 0,
    graph_execs: 0,
    allocations: 0,
    modules: 0,
    functions: 0,
    executors: 0,
    device_bytes: 0L,
    pending_work: 0,
    cleanup_failures: 0,
  }
}

///|
fn FusedProbeResourceBalance::is_zero(self : FusedProbeResourceBalance) -> Bool {
  self.contexts == 0 &&
  self.streams == 0 &&
  self.events == 0 &&
  self.graphs == 0 &&
  self.graph_execs == 0 &&
  self.allocations == 0 &&
  self.modules == 0 &&
  self.functions == 0 &&
  self.executors == 0 &&
  self.device_bytes == 0L &&
  self.pending_work == 0 &&
  self.cleanup_failures == 0
}

///|
fn FusedProbeResourceBalance::render(
  self : FusedProbeResourceBalance,
) -> String {
  "context\{self.contexts},stream\{self.streams},event\{self.events},graph\{self.graphs},graph_exec\{self.graph_execs},allocation\{self.allocations},module\{self.modules},function\{self.functions},executor\{self.executors},device_bytes\{self.device_bytes},pending\{self.pending_work},cleanup\{self.cleanup_failures}"
}
