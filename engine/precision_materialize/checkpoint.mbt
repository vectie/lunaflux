///|
pub(all) suberror PackedCheckpointError {
  MissingTensor(String)
  TensorStorageMismatch(String)
  InvalidPlaneNames
  InvalidDestinationMapping
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
  priv transfer_session : @streaming.StreamingSafetensorsTransferSession?
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
  { inspection, locations, transfer_session: None, }
}

///|
/// Scope authenticated open shard reuse to one synchronous startup upload.
/// The callback borrows the indexed checkpoint; no file effect reaches token
/// execution. Closing this scope retires every cached handle on success/error.
pub fn[T] PackedCheckpoint::with_transfer_session(
  self : Self,
  root : @approved_fs.ApprovedRoot,
  limits : @streaming.StreamingSafetensorsLimits,
  body : (Self) -> T raise,
) -> T raise {
  let session = @streaming.StreamingSafetensorsTransferSession::new(
    root,
    self.inspection,
    limits,
  )
  defer session.close()
  body({
    inspection: self.inspection,
    locations: self.locations,
    transfer_session: Some(session),
  })
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
  byte_alias? : String,
) -> CheckpointLocation raise PackedCheckpointError {
  guard self.locations.get(name) is Some(location) else {
    raise MissingTensor(name)
  }
  let tensor = location.metadata
  if (tensor.dtype().tag() != tag && byte_alias != Some(tensor.dtype().tag())) ||
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
    byte_alias?=if rep.element() == E2M1 { Some("I8") } else { None },
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
          byte_alias?=if rep.scale_element() == Ue8M0 {
            Some("F8_E8M0")
          } else {
            None
          },
        ),
      )
    }
  }
  let layout = @precision.packed_buffer_layout(rep) catch {
    _ => raise BufferSizeOverflow
  }
  let scales_offset = layout.scales_offset().unwrap_or(0L)
  let global_scale_offset = layout.global_scale_offset().unwrap_or(0L)
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
    buffer_bytes: layout.buffer_bytes(),
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
  self.transfer_weights_to(
    root,
    weights,
    destinations=Array::makei(weights.length(), i => (i, 0L)),
    destination_bytes=weights.map(weight => weight.buffer_bytes()),
    scratch~,
    limits~,
    consume~,
  )
}

///|
/// The same bounded transfer into a planned bank or segmented arena. Each
/// weight gets an explicit destination ordinal and base offset; plane offsets
/// remain the shared precision layout. No temporary whole-bank host buffer.
pub fn PackedCheckpoint::transfer_weights_to(
  self : Self,
  root : @approved_fs.ApprovedRoot,
  weights : ArrayView[PackedCheckpointWeight],
  destinations~ : ArrayView[(Int, Int64)],
  destination_bytes~ : ArrayView[Int64],
  scratch~ : FixedArray[Byte],
  limits~ : @streaming.StreamingSafetensorsLimits,
  consume~ : (Int, Int64, FixedArray[Byte], Int) -> Bool,
) -> Unit raise PackedCheckpointError {
  if destinations.length() != weights.length() {
    raise InvalidDestinationMapping
  }
  let requests = []
  for index, weight in weights {
    let (destination, base) = destinations[index]
    if destination < 0 ||
      base < 0L ||
      base > 9223372036854775807L - weight.buffer_bytes {
      raise InvalidDestinationMapping
    }
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
    requests.push(transfer_plane(weight.payload, destination, base))
    if weight.scales is Some(location) {
      requests.push(
        transfer_plane(location, destination, base + weight.scales_offset),
      )
    }
    if weight.global_scale is Some(location) {
      requests.push(
        transfer_plane(location, destination, base + weight.global_scale_offset),
      )
    }
  }
  try {
    match self.transfer_session {
      Some(session) =>
        session.transfer_ranges(
          root,
          requests~,
          destination_bytes~,
          scratch~,
          limits~,
          consume~,
        )
      None =>
        @streaming.transfer_ranges(
          root,
          self.inspection,
          requests~,
          destination_bytes~,
          scratch~,
          limits~,
          consume~,
        )
    }
  } catch {
    error => raise Streaming(error)
  }
}
