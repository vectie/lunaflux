///|
priv struct LoadedResource[T] {
  resource : T
  layout : DeviceWeightLayout
}

///|
priv enum ResourcePreparation[T] {
  ResourceReady(LoadedResource[T])
  ResourceCleanupRequired(FailedResourcePreparation[T])
}

///|
priv struct FailedResourcePreparation[T] {
  resource : T
  cause : DeviceWeightError
  stage : DeviceWeightFailureStage
  cleanup_failure : @device.DeviceError
}

///|
fn[T] resource_failure(
  resource : T,
  cause : DeviceWeightError,
  stage : DeviceWeightFailureStage,
  close : (T) -> Unit raise @device.DeviceError,
) -> ResourcePreparation[T] raise DeviceWeightError {
  try close(resource) catch {
    cleanup_failure =>
      ResourceCleanupRequired({ resource, cause, stage, cleanup_failure })
  } noraise {
    _ => raise cause
  }
}
