///|
async fn open_regular_file(
  path : StringView,
) -> @fs.File raise DeviceWeightError {
  let path_kind = @fs.kind(path, follow_symlink=false) catch {
    _ => raise FileUnavailable
  }
  if path_kind != Regular {
    raise UnsupportedFileType
  }
  let file = @fs.open(path, mode=ReadOnly) catch { _ => raise FileUnavailable }
  if file.kind() != Regular {
    file.close()
    raise UnsupportedFileType
  }
  file
}

///|
async fn prepare_open_file(
  file : @fs.File,
  model : @spec.LlamaModelMetadata,
  limits : DeviceWeightFileLimits,
  alignment_bytes : Int64,
  max_arena_bytes : Int64,
) -> PreparedStreamingLoad raise DeviceWeightError {
  let initial_stamp = source_stamp(file) catch { _ => raise FileUnavailable }
  prepare_streaming_load(
    model,
    initial_stamp,
    limits,
    alignment_bytes,
    max_arena_bytes,
    (position, count) => file.read_exactly_at(count, position~),
    () => source_stamp(file),
  )
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
/// Inspects one dense-Llama safetensors file without opening a device resource.
///
/// The caller supplies a locator governed by a deployment-owned stable,
/// read-only model root. This API does not establish that root authority. It
/// opens one regular file read-only, validates the bounded header and exact
/// tensor vocabulary, plans every final arena region, hashes the complete file
/// against the model identity, and rechecks the same open handle's metadata.
///
/// MoonBit's portable file API does not provide atomic `openat` plus no-follow,
/// so the path-kind check and handle open retain a documented TOCTOU window.
/// The opened-handle kind check and complete content digest remain authoritative
/// for the bytes admitted by this call.
pub async fn inspect_file(
  model : @spec.LlamaModelMetadata,
  path : StringView,
  limits~ : DeviceWeightFileLimits,
  alignment_bytes~ : Int64,
  max_arena_bytes~ : Int64,
) -> DeviceWeightFileInspection raise DeviceWeightError {
  let file = open_regular_file(path)
  defer file.close()
  let prepared = prepare_open_file(
    file, model, limits, alignment_bytes, max_arena_bytes,
  )
  inspection_from_prepared(
    model, limits, alignment_bytes, max_arena_bytes, prepared,
  )
}

///|
/// Loads a previously inspected weight file through a freshly opened handle.
///
/// The caller must again supply a locator governed by the deployment-owned
/// stable, read-only root; this API does not establish that root authority and
/// retains the same documented path-kind/open TOCTOU window as `inspect_file`.
///
/// Filesystem metadata from inspection is never trusted across calls. Before
/// allocating device memory this function completely re-admits and re-hashes
/// the current bytes, then requires their semantic layout and transfer regions
/// to equal the immutable inspection evidence. Transfer performs a second
/// complete hash and same-handle metadata check while copying bounded chunks.
pub async fn load_inspected_file(
  context : @device.Context,
  inspection : DeviceWeightFileInspection,
  path : StringView,
) -> DeviceWeightPreparation raise DeviceWeightError {
  let file = open_regular_file(path)
  defer file.close()
  let initial_stamp = source_stamp(file) catch { _ => raise FileUnavailable }
  let loaded = load_inspected_streaming_with_resource(
    inspection,
    initial_stamp,
    (position, count) => file.read_exactly_at(count, position~),
    () => source_stamp(file),
    arena_bytes => context.allocate(arena_bytes),
    (allocation, bytes, destination_offset, byte_count) => {
      allocation.copy_from_host(bytes, 0L, destination_offset, byte_count)
    },
    allocation => allocation.close(),
  )
  device_weight_preparation(loaded)
}

///|
/// Streams a caller-selected dense-Llama safetensors file into one final
/// aligned device arena without retaining a model-sized host snapshot.
///
/// Admission rejects non-regular paths, opens the file read-only, validates
/// its bounded header and exact Llama tensor vocabulary, plans the complete
/// arena, and verifies the model identity's SHA-256 digest before opening the
/// device allocation. A second bounded pass hashes the exact bytes used while
/// copying each tensor directly to its final region. Same-size mutation is
/// therefore rejected and the partial allocation is closed deterministically.
///
/// The deployment contract still requires an approved read-only, stable model
/// mount. MoonBit's portable file API does not expose an atomic `openat` plus
/// no-follow primitive, so the loader performs both a no-follow path-kind check
/// and an opened-handle kind check but does not claim stronger path-race
/// exclusion. Digest validation remains authoritative for admitted bytes.
///
/// `WeightsReady` transfers complete weight ownership to the caller.
/// `WeightsCleanupRequired` retains an incomplete allocation after both the
/// primary load and first close fail; callers must retry its close before the
/// context can be released.
pub async fn load_file(
  context : @device.Context,
  model : @spec.LlamaModelMetadata,
  path : StringView,
  limits~ : DeviceWeightFileLimits,
  alignment_bytes~ : Int64,
  max_arena_bytes~ : Int64,
) -> DeviceWeightPreparation raise DeviceWeightError {
  let file = open_regular_file(path)
  defer file.close()
  let initial_stamp = source_stamp(file) catch { _ => raise FileUnavailable }
  let loaded = load_streaming_with_resource(
    model,
    initial_stamp,
    limits,
    alignment_bytes,
    max_arena_bytes,
    (position, count) => file.read_exactly_at(count, position~),
    () => source_stamp(file),
    arena_bytes => context.allocate(arena_bytes),
    (allocation, bytes, destination_offset, byte_count) => {
      allocation.copy_from_host(bytes, 0L, destination_offset, byte_count)
    },
    allocation => allocation.close(),
  )
  device_weight_preparation(loaded)
}

///|
async fn source_stamp(file : @fs.File) -> SourceStamp {
  let file_bytes = file.size()
  let (modified_seconds, modified_nanoseconds) = file.mtime()
  let (changed_seconds, changed_nanoseconds) = file.ctime()
  {
    file_bytes,
    modified_seconds,
    modified_nanoseconds,
    changed_seconds,
    changed_nanoseconds,
  }
}
