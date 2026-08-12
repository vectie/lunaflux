///|
priv struct LoadedResource[T] {
  resource : T
  layout : DeviceWeightLayout
}

///|
fn[T] close_after_failure(
  resource : T,
  stage : DeviceWeightFailureStage,
  close : (T) -> Unit raise @device.DeviceError,
) -> Unit raise DeviceWeightError {
  close(resource) catch {
    error => raise CleanupFailed(stage~, error)
  }
}

///|
fn[T] close_owned_after_failure(
  owned : Ref[T?],
  stage : DeviceWeightFailureStage,
  close : (T) -> Unit raise @device.DeviceError,
) -> Unit raise DeviceWeightError {
  match owned.val {
    Some(resource) => close_after_failure(resource, stage, close)
    None => ()
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
) -> LoadedResource[T] raise DeviceWeightError {
  let owned : Ref[T?] = Ref(None)
  let visited = @materialize.visit(
    bindings,
    file,
    limits=materialization_limits,
    chunk => {
      let reference = chunk.reference().as_int()
      guard layout.region(chunk.reference()) is Some(region) else {
        close_owned_after_failure(owned, TensorValidation, close)
        raise UnexpectedTensor(reference~)
      }
      if region.byte_length() != chunk.tensor().byte_length() {
        close_owned_after_failure(owned, TensorValidation, close)
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
        error => {
          close_after_failure(resource, Transfer, close)
          raise DeviceFailed(error)
        }
      }
    },
  )
  match visited {
    Err(error) => {
      // `visit` completes validation before invoking its visitor. Consequently
      // malformed artifacts reach this branch with no opened resource.
      close_owned_after_failure(owned, MaterializationResult, close)
      raise MaterializationFailed(error)
    }
    Ok(report) => {
      guard owned.val is Some(resource) else { raise EmptyBindings }
      if report.identity() != layout.identity() ||
        report.total_bytes() != layout.materialized_bytes() ||
        report.length() != layout.length() {
        close_after_failure(resource, MaterializationResult, close)
        raise TensorLengthMismatch(reference=0)
      }
      { resource, layout }
    }
  }
}

///|
/// Validates a complete safetensors envelope before the first device
/// allocation, then copies each tensor from the original immutable file
/// directly into its final aligned arena region.
///
/// Any failure after allocation attempts deterministic cleanup before it is
/// returned. A cleanup failure retains the bounded primary failure stage and
/// supersedes the primary error; the native finalizer remains a defensive
/// retry rather than the normal ownership mechanism.
///
/// The caller owns the returned `DeviceWeights` and must close it before its
/// context.
pub fn load(
  context : @device.Context,
  bindings : @llama_weights.LlamaWeightBindings,
  file : Bytes,
  materialization_limits~ : @materialize.MaterializationLimits,
  alignment_bytes~ : Int64,
  max_arena_bytes~ : Int64,
) -> DeviceWeights raise DeviceWeightError {
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
  { allocation: loaded.resource, layout: loaded.layout }
}
