///|
/// Streams an approved dense-Llama safetensors file directly into one final
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
  let path_kind = @fs.kind(path, follow_symlink=false) catch {
    _ => raise FileUnavailable
  }
  if path_kind != Regular {
    raise UnsupportedFileType
  }
  let file = @fs.open(path, mode=ReadOnly) catch { _ => raise FileUnavailable }
  defer file.close()
  if file.kind() != Regular {
    raise UnsupportedFileType
  }
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
async fn source_stamp(file : @fs.File) -> SourceStamp {
  let file_bytes = file.size()
  let (modified_seconds, modified_nanoseconds) = file.mtime()
  { file_bytes, modified_seconds, modified_nanoseconds }
}
