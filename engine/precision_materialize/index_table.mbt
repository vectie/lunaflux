///|
/// Exact checkpoint storage for a discrete index table, not floating-point
/// numerical storage. Binding never reads payloads or allocates a host arena.
pub struct CheckpointI64Table {
  priv location : CheckpointLocation
  priv rows : Int
  priv columns : Int
  priv input_bytes : Int64
}

///|
pub fn PackedCheckpoint::bind_i64_table(
  self : Self,
  payload_name~ : String,
  rows~ : Int,
  columns~ : Int,
) -> CheckpointI64Table raise PackedCheckpointError {
  guard rows > 0 &&
    columns > 0 &&
    rows.to_int64() <= 9223372036854775807L / columns.to_int64() / 8L else {
    raise TensorStorageMismatch(payload_name)
  }
  let input_bytes = rows.to_int64() * columns.to_int64() * 8L
  {
    location: self.checkpoint_plane(
      payload_name,
      "I64",
      [rows.to_int64(), columns.to_int64()],
      input_bytes,
    ),
    rows,
    columns,
    input_bytes,
  }
}

///|
pub fn CheckpointI64Table::geometry(self : Self) -> (Int, Int) {
  (self.rows, self.columns)
}

///|
pub fn CheckpointI64Table::input_bytes(self : Self) -> Int64 {
  self.input_bytes
}

///|
pub fn CheckpointI64Table::i32_bytes(self : Self) -> Int64 {
  self.input_bytes / 2L
}

///|
/// Synchronous bounded startup reads. Sink offsets/lengths refer to the I64
/// source stream, not a device destination. A narrowing sink budgets its I32
/// allocation separately and cannot retain the reused scratch chunk.
pub fn PackedCheckpoint::transfer_i64_tables(
  self : Self,
  root : @approved_fs.ApprovedRoot,
  tables : ArrayView[CheckpointI64Table],
  scratch~ : FixedArray[Byte],
  limits~ : @streaming.StreamingSafetensorsLimits,
  consume~ : (Int, Int64, FixedArray[Byte], Int) -> Bool,
) -> Unit raise PackedCheckpointError {
  let requests = []
  for index, table in tables {
    let location = table.location
    guard self.locations.get(location.metadata.name()) is Some(current) &&
      current.shard == location.shard &&
      current.tensor == location.tensor &&
      current.metadata == location.metadata else {
      raise TensorStorageMismatch(location.metadata.name())
    }
    requests.push(transfer_plane(location, index, 0L))
  }
  let destination_bytes = tables.map(table => table.input_bytes)
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
