///|
pub(all) suberror PackedCheckpointError {
  MissingTensor(String)
  TensorStorageMismatch(String)
  InvalidPlaneNames
  BufferSizeOverflow
  Streaming(@streaming.StreamingSafetensorsError)
} derive(Debug, Eq)

///|
pub extend PackedCheckpointError with Eq::{equal, not_equal}

///|
pub extend PackedCheckpointError with @debug.Debug::{to_repr}

///|
priv struct CheckpointLocation {
  shard : Int
  tensor : Int
  metadata : @streaming.StreamingSafetensorsTensorMetadata
}

///|
/// Startup index only. Packed payloads stay in their original checkpoint;
/// neither this index nor the weight bindings allocate a host weight arena.
pub struct PackedCheckpoint {
  priv inspection : @streaming.StreamingSafetensorsInspection
  priv locations : Map[String, CheckpointLocation]
}

///|
pub fn PackedCheckpoint::new(
  inspection : @streaming.StreamingSafetensorsInspection,
) -> Self {
  let locations = Map([])
  for shard in 0..<inspection.shard_count() {
    let metadata = inspection.shard(shard).unwrap()
    for tensor in 0..<metadata.tensor_count() {
      let entry = metadata.tensor(tensor).unwrap()
      locations.set(entry.name(), { shard, tensor, metadata: entry, })
    }
  }
  { inspection, locations, }
}

///|
/// The offsets describe one compact destination allocation. Device lowering
/// can use separate payload/scale pointers without expanding the weight bank.
pub struct PackedCheckpointWeight {
  priv representation : @format.Representation
  priv payload : CheckpointLocation
  priv scales : CheckpointLocation?
  priv global_scale : CheckpointLocation?
  priv scales_offset : Int64
  priv global_scale_offset : Int64
  priv buffer_bytes : Int64
}

///|
fn PackedCheckpoint::checkpoint_plane(
  self : PackedCheckpoint,
  name : String,
  tag : String,
  shape : ArrayView[Int64],
  bytes : Int64,
) -> CheckpointLocation raise PackedCheckpointError {
  guard self.locations.get(name) is Some(location) else {
    raise MissingTensor(name)
  }
  let tensor = location.metadata
  if tensor.dtype().tag() != tag ||
    tensor.shape() != ReadOnlyArray::from_array(shape) ||
    tensor.byte_count() != bytes {
    raise TensorStorageMismatch(name)
  }
  location
}

///|
/// Pure physical binding. Checkpoint names are supplied by the model adapter;
/// shape, packing, scale grids, and byte offsets use shared precision semantics.
pub fn PackedCheckpoint::bind_weight(
  self : Self,
  representation~ : @format.Representation,
  payload_name~ : String,
  scales_name? : String,
  global_scale_name? : String,
) -> PackedCheckpointWeight raise PackedCheckpointError {
  let rep = representation
  if (scales_name is Some(_)) != (rep.scale_bytes() > 0L) ||
    (global_scale_name is Some(_)) != rep.has_global_scale() ||
    scales_name == Some(payload_name) ||
    global_scale_name == Some(payload_name) ||
    (scales_name is Some(_) && scales_name == global_scale_name) {
    raise InvalidPlaneNames
  }
  let (rows, columns) = rep.shape()
  let (payload_tag, payload_columns) = match rep.element() {
    Bf16 => ("BF16", columns.to_int64())
    Fp16 => ("F16", columns.to_int64())
    F32 => ("F32", columns.to_int64())
    E4M3 => ("F8_E4M3", columns.to_int64())
    E5M2 => ("F8_E5M2", columns.to_int64())
    Signed8 => ("I8", columns.to_int64())
    Signed4 | E2M1 => ("U8", rep.row_bytes())
  }
  let payload = self.checkpoint_plane(
    payload_name,
    payload_tag,
    [rows.to_int64(), payload_columns],
    rep.payload_bytes(),
  )
  let (scale_rows, scale_columns) = rep.scale_shape()
  let scales = match scales_name {
    None => None
    Some(name) => {
      let tag = match rep.scale_element() {
        F32 => "F32"
        E4M3 => "F8_E4M3"
        Ue8M0 => "U8"
        Absent => raise InvalidPlaneNames
      }
      Some(
        self.checkpoint_plane(
          name,
          tag,
          [scale_rows.to_int64(), scale_columns.to_int64()],
          rep.scale_bytes(),
        ),
      )
    }
  }
  if rep.payload_bytes() > 9223372036854775807L - rep.scale_bytes() - 7L {
    raise BufferSizeOverflow
  }
  let scales_offset = rep.payload_bytes()
  // F32 scale planes need natural alignment, including odd-width INT4 rows.
  let scales_offset = if rep.scale_element() == F32 {
    (scales_offset + 3L) / 4L * 4L
  } else {
    scales_offset
  }
  let end = scales_offset + rep.scale_bytes()
  let global_scale_offset = (end + 3L) / 4L * 4L
  let global_scale = match global_scale_name {
    None => None
    Some(name) => Some(self.checkpoint_plane(name, "F32", [], 4L))
  }
  {
    representation: rep,
    payload,
    scales,
    global_scale,
    scales_offset,
    global_scale_offset,
    buffer_bytes: if rep.has_global_scale() {
      global_scale_offset + 4L
    } else {
      end
    },
  }
}

///|
pub fn PackedCheckpointWeight::representation(
  self : Self,
) -> @format.Representation {
  self.representation
}

///|
pub fn PackedCheckpointWeight::scales_offset(self : Self) -> Int64? {
  self.scales.map(_ => self.scales_offset)
}

///|
pub fn PackedCheckpointWeight::global_scale_offset(self : Self) -> Int64? {
  self.global_scale.map(_ => self.global_scale_offset)
}

///|
pub fn PackedCheckpointWeight::buffer_bytes(self : Self) -> Int64 {
  self.buffer_bytes
}

///|
fn transfer_plane(
  location : CheckpointLocation,
  destination : Int,
  offset : Int64,
) -> @streaming.StreamingSafetensorsTransfer raise PackedCheckpointError {
  @streaming.StreamingSafetensorsTransfer::new(
    shard_ordinal=location.shard,
    tensor_ordinal=location.tensor,
    tensor_byte_offset=0L,
    destination_ordinal=destination,
    destination_offset=offset,
    byte_count=location.metadata.byte_count(),
  ) catch {
    error => raise Streaming(error)
  }
}

///|
/// Bounded startup effect: one destination per weight, one reusable scratch
/// buffer for all planes. The sink synchronously copies/consumes each chunk.
/// The caller retains allocation ownership on partial failure. Padding between
/// planes is untouched and is never part of a kernel's logical operands.
pub fn PackedCheckpoint::transfer_weights(
  self : Self,
  root : @approved_fs.ApprovedRoot,
  weights : ArrayView[PackedCheckpointWeight],
  scratch~ : FixedArray[Byte],
  limits~ : @streaming.StreamingSafetensorsLimits,
  consume~ : (Int, Int64, FixedArray[Byte], Int) -> Bool,
) -> Unit raise PackedCheckpointError {
  let requests = []
  let destination_bytes = []
  for destination, weight in weights {
    // Prevent mixing incompatible ordinal mappings from different indexes.
    // File authentication remains the shared reader's startup responsibility.
    for location in [Some(weight.payload), weight.scales, weight.global_scale] {
      if location is Some(location) {
        guard self.locations.get(location.metadata.name()) is Some(current) &&
          current.shard == location.shard &&
          current.tensor == location.tensor &&
          current.metadata == location.metadata else {
          raise TensorStorageMismatch(location.metadata.name())
        }
      }
    }
    destination_bytes.push(weight.buffer_bytes)
    requests.push(transfer_plane(weight.payload, destination, 0L))
    if weight.scales is Some(location) {
      requests.push(transfer_plane(location, destination, weight.scales_offset))
    }
    if weight.global_scale is Some(location) {
      requests.push(
        transfer_plane(location, destination, weight.global_scale_offset),
      )
    }
  }
  @streaming.transfer_ranges(
    root,
    self.inspection,
    requests~,
    destination_bytes~,
    scratch~,
    limits~,
    consume~,
  ) catch {
    error => raise Streaming(error)
  }
}
