///|
suberror DeviceWorkerChildError {
  StartupFailed
  CleanEof
} derive(Debug, Eq)

///|
priv enum ChildBootstrap[T, F] {
  BootstrapReady(T)
  BootstrapCleanup(F)
}

///|
priv struct ChildSteadyBuffers {
  plan_owner : @worker_wire.PlanFrameBuffer
  completion_owner : @worker_wire.CompletionFrameBuffer
  input : FixedArray[Byte]
  output : FixedArray[Byte]
  graph_output : FixedArray[Byte]
}
