///|
fn open_weight_file(
  root : @approved_fs.ApprovedRoot,
  locator : @approved_fs.ApprovedRelativeLocator,
) -> @approved_fs.ApprovedFile raise DeviceWeightError {
  root.open_file(locator) catch {
    UnsupportedType(File) => raise UnsupportedFileType
    _ => raise FileUnavailable
  }
}

///|
fn read_weight_range(
  file : @approved_fs.ApprovedFile,
  destination : FixedArray[Byte],
  destination_offset : Int,
  position : Int64,
  byte_count : Int,
) -> Unit raise DeviceWeightError {
  file.read_exactly_at(destination, destination_offset~, byte_count~, position~) catch {
    Truncated => raise FileTruncated
    _ => raise FileUnavailable
  }
}

///|
fn weight_file_stamp(
  file : @approved_fs.ApprovedFile,
) -> @approved_fs.ApprovedFileStamp raise DeviceWeightError {
  file.stamp() catch {
    _ => raise FileUnavailable
  }
}

///|
fn close_weight_file(
  file : @approved_fs.ApprovedFile,
) -> Unit raise DeviceWeightError {
  file.close() catch {
    _ => raise FileCloseFailed
  }
}

///|
fn[T] finish_weight_file_body(
  outcome : Result[T, DeviceWeightError],
  close_file : () -> Unit raise DeviceWeightError,
) -> T raise DeviceWeightError {
  match outcome {
    Err(primary) => {
      close_file() catch {
        _ => ()
      }
      raise primary
    }
    Ok(value) => {
      close_file()
      value
    }
  }
}

///|
fn[T] with_weight_file(
  root : @approved_fs.ApprovedRoot,
  locator : @approved_fs.ApprovedRelativeLocator,
  body : (@approved_fs.ApprovedFile) -> T raise DeviceWeightError,
) -> T raise DeviceWeightError {
  let file = open_weight_file(root, locator)
  let outcome : Result[T, DeviceWeightError] = Ok(body(file)) catch {
    error => Err(error)
  }
  finish_weight_file_body(outcome, () => close_weight_file(file))
}

///|
fn device_weight_preparation(
  loaded : ResourcePreparation[@device.Allocation],
) -> DeviceWeightPreparation {
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

///|
fn[T] finish_weight_file_load(
  outcome : Result[ResourcePreparation[T], DeviceWeightError],
  close_file : () -> Unit raise DeviceWeightError,
  close_resource : (T) -> Unit raise @device.DeviceError,
) -> ResourcePreparation[T] raise DeviceWeightError {
  match outcome {
    Err(primary) => {
      close_file() catch {
        _ => ()
      }
      raise primary
    }
    Ok(loaded) =>
      try close_file() catch {
        close_failure =>
          match loaded {
            // The primary load failure already owns the retryable allocation;
            // a consumed file descriptor adds no remaining cleanup authority.
            ResourceCleanupRequired(_) => loaded
            ResourceReady(ready) =>
              resource_failure(
                ready.resource,
                close_failure,
                SourceClose,
                close_resource,
              )
          }
      } noraise {
        _ => loaded
      }
  }
}

///|
fn with_weight_file_load(
  root : @approved_fs.ApprovedRoot,
  locator : @approved_fs.ApprovedRelativeLocator,
  body : (@approved_fs.ApprovedFile) -> ResourcePreparation[@device.Allocation] raise DeviceWeightError,
) -> DeviceWeightPreparation raise DeviceWeightError {
  let file = open_weight_file(root, locator)
  let outcome : Result[
    ResourcePreparation[@device.Allocation],
    DeviceWeightError,
  ] = Ok(body(file)) catch { error => Err(error) }
  device_weight_preparation(
    finish_weight_file_load(outcome, () => close_weight_file(file), allocation => {
      allocation.close()
    }),
  )
}

///|
/// Inspects one dense-Llama safetensors descendant without opening a device
/// resource.
///
/// `root` is an independently approved pinned directory capability and
/// `locator` must be a strict relative descendant. The loader opens the file
/// component-by-component without following symlinks, validates the bounded
/// header and exact tensor vocabulary, plans every final arena region, hashes
/// the complete pinned file against the model identity, and rechecks metadata
/// from that same handle. The validated locator is retained privately in the
/// returned inert evidence.
pub fn inspect_file(
  root : @approved_fs.ApprovedRoot,
  model : @spec.LlamaModelMetadata,
  locator : @approved_fs.ApprovedRelativeLocator,
  limits~ : DeviceWeightFileLimits,
  alignment_bytes~ : Int64,
  max_arena_bytes~ : Int64,
) -> DeviceWeightFileInspection raise DeviceWeightError {
  with_weight_file(root, locator, file => {
    let initial_stamp = weight_file_stamp(file)
    let prepared = prepare_streaming_load(
      model,
      initial_stamp,
      initial_stamp.byte_count(),
      limits,
      alignment_bytes,
      max_arena_bytes,
      (destination, destination_offset, position, byte_count) => {
        read_weight_range(
          file, destination, destination_offset, position, byte_count,
        )
      },
      () => weight_file_stamp(file),
    )
    inspection_from_prepared(
      model, locator, limits, alignment_bytes, max_arena_bytes, prepared,
    )
  })
}

///|
/// Loads a previously inspected weight descendant through a fresh pinned
/// handle beneath `root`.
///
/// The private inspected locator cannot be substituted by the caller.
/// Filesystem metadata is never trusted across calls: before allocating device
/// memory this function completely re-admits and re-hashes the current bytes,
/// then requires their semantic layout and transfer regions to equal the
/// immutable inspection evidence. Transfer performs another complete hash and
/// same-handle metadata check while copying bounded reusable host chunks.
/// A terminal file-close failure closes any otherwise-ready allocation; if
/// that allocation close also fails, cleanup authority is returned at
/// `SourceClose` rather than publishing readiness.
pub fn load_inspected_file(
  context : @device.Context,
  root : @approved_fs.ApprovedRoot,
  inspection : DeviceWeightFileInspection,
) -> DeviceWeightPreparation raise DeviceWeightError {
  with_weight_file_load(root, inspection.locator, file => {
    let initial_stamp = weight_file_stamp(file)
    let loaded = load_inspected_streaming_with_resource(
      inspection,
      initial_stamp,
      initial_stamp.byte_count(),
      (destination, destination_offset, position, byte_count) => {
        read_weight_range(
          file, destination, destination_offset, position, byte_count,
        )
      },
      () => weight_file_stamp(file),
      arena_bytes => context.allocate(arena_bytes),
      (allocation, source, source_offset, destination_offset, byte_count) => {
        context.copy_from_fixed_host(
          allocation,
          source,
          source_offset.to_int64(),
          destination_offset,
          byte_count,
        )
      },
      allocation => allocation.close(),
    )
    loaded
  })
}

///|
/// Streams one approved dense-Llama safetensors descendant into a final
/// aligned device arena without retaining a model-sized host snapshot.
///
/// Admission opens the strict relative locator beneath `root`, validates the
/// complete file and model digest before allocating the arena, then performs a
/// second bounded pass that hashes the exact bytes copied through one reusable
/// fixed host buffer. Namespace replacement cannot redirect either pinned
/// handle; digest and same-handle stamp checks remain authoritative for bytes.
///
/// `WeightsReady` transfers complete weight ownership to the caller.
/// `WeightsCleanupRequired` retains an incomplete allocation when either a
/// primary load failure or terminal source-file close failure is followed by
/// allocation-close failure. Callers must retry its close before the context
/// can be released.
pub fn load_file(
  context : @device.Context,
  root : @approved_fs.ApprovedRoot,
  model : @spec.LlamaModelMetadata,
  locator : @approved_fs.ApprovedRelativeLocator,
  limits~ : DeviceWeightFileLimits,
  alignment_bytes~ : Int64,
  max_arena_bytes~ : Int64,
) -> DeviceWeightPreparation raise DeviceWeightError {
  with_weight_file_load(root, locator, file => {
    let initial_stamp = weight_file_stamp(file)
    let loaded = load_streaming_with_resource(
      model,
      initial_stamp,
      initial_stamp.byte_count(),
      limits,
      alignment_bytes,
      max_arena_bytes,
      (destination, destination_offset, position, byte_count) => {
        read_weight_range(
          file, destination, destination_offset, position, byte_count,
        )
      },
      () => weight_file_stamp(file),
      arena_bytes => context.allocate(arena_bytes),
      (allocation, source, source_offset, destination_offset, byte_count) => {
        context.copy_from_fixed_host(
          allocation,
          source,
          source_offset.to_int64(),
          destination_offset,
          byte_count,
        )
      },
      allocation => allocation.close(),
    )
    loaded
  })
}
