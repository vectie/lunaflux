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

///|
fn[T] load_with_resource(
  bindings : @llama_weights.LlamaWeightBindings,
  file : Bytes,
  layout : DeviceWeightLayout,
  materialization_limits : @materialize.MaterializationLimits,
  open : () -> T raise @device.DeviceError,
  copy : (T, Int64, Int64, Int64) -> Unit raise @device.DeviceError,
  close : (T) -> Unit raise @device.DeviceError,
) -> ResourcePreparation[T] raise DeviceWeightError {
  let owned : Ref[T?] = Ref(None)
  let visited = Ok(
    @materialize.visit(bindings, file, limits=materialization_limits, chunk => {
      let reference = chunk.reference().as_int()
      guard layout.region(chunk.reference()) is Some(region) else {
        raise UnexpectedTensor(reference~)
      }
      if region.byte_length() != chunk.tensor().byte_length() {
        raise TensorLengthMismatch(reference~)
      }
      let resource = match owned.val {
        Some(resource) => resource
        None => {
          let resource = open() catch { error => raise DeviceFailed(error) }
          owned.val = Some(resource)
          resource
        }
      }
      copy(
        resource,
        chunk.file_offset_bytes(),
        region.offset_bytes(),
        region.byte_length(),
      ) catch {
        error => raise DeviceFailed(error)
      }
    }),
  ) catch {
    error => Err(error)
  }
  match visited {
    Err(error) =>
      match owned.val {
        Some(resource) => {
          let stage = match error {
            DeviceFailed(_) => Transfer
            UnexpectedTensor(_) | TensorLengthMismatch(_) => TensorValidation
            _ => MaterializationResult
          }
          resource_failure(resource, error, stage, close)
        }
        None => raise error
      }
    Ok(result) =>
      match result {
        Err(error) =>
          // Validation is complete before visiting, so no resource is open.
          raise MaterializationFailed(error)
        Ok(report) => {
          guard owned.val is Some(resource) else { raise EmptyBindings }
          if report.identity() != layout.identity() ||
            report.total_bytes() != layout.materialized_bytes() ||
            report.length() != layout.length() {
            return resource_failure(
              resource,
              TensorLengthMismatch(reference=0),
              MaterializationResult,
              close,
            )
          }
          ResourceReady({ resource, layout })
        }
      }
  }
}

///|
/// Validates a complete safetensors envelope before the first device
/// allocation, then copies each tensor from the original immutable file
/// directly into its final aligned arena region.
///
/// Any failure after allocation attempts deterministic cleanup. If that close
/// also fails, the result retains the allocation, exact primary cause/stage,
/// and latest cleanup error for explicit retry. The native finalizer remains
/// a defensive fallback rather than the normal ownership mechanism.
///
/// `WeightsReady` transfers complete weight ownership to the caller.
/// `WeightsCleanupRequired` must be retried before closing the context.
pub fn load(
  context : @device.Context,
  bindings : @llama_weights.LlamaWeightBindings,
  file : Bytes,
  materialization_limits~ : @materialize.MaterializationLimits,
  alignment_bytes~ : Int64,
  max_arena_bytes~ : Int64,
) -> DeviceWeightPreparation raise DeviceWeightError {
  let layout = plan_layout(bindings, alignment_bytes~, max_arena_bytes~)
  let loaded = load_with_resource(
    bindings,
    file,
    layout,
    materialization_limits,
    () => context.allocate(layout.arena_bytes()),
    (allocation, source_offset, destination_offset, byte_count) => {
      allocation.copy_from_host(
        file, source_offset, destination_offset, byte_count,
      )
    },
    allocation => allocation.close(),
  )
  match loaded {
    ResourceReady(ready) =>
      WeightsReady({ allocation: ready.resource, layout: ready.layout })
    ResourceCleanupRequired(failed) =>
      WeightsCleanupRequired({
        allocation: failed.resource,
        cause: failed.cause,
        stage: failed.stage,
        cleanup_failure: failed.cleanup_failure,
        open: true,
      })
  }
}
