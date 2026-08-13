///|
suberror DeviceWorkerChildError {
  StartupFailed
} derive(Debug, Eq)

///|
priv enum ChildBootstrap[T, F] {
  BootstrapReady(T)
  BootstrapCleanup(F)
}
