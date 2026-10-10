///|
/// Explicit startup-only cache of pinned inspected shard handles. A cache
/// miss uses the same file checks as transfer_ranges; reuse checks the
/// same-handle stamp before and after each transfer. No host weight arena.
pub struct StreamingSafetensorsTransferSession {
  priv root : @approved_fs.ApprovedRoot
  priv inspection : StreamingSafetensorsInspection
  priv files : Array[@approved_fs.ApprovedFile?]
  priv mut opened : Int
  priv mut closed : Bool
  priv mut failed : Bool
  priv mut busy : Bool
}

///|
pub fn StreamingSafetensorsTransferSession::new(
  root : @approved_fs.ApprovedRoot,
  inspection : StreamingSafetensorsInspection,
  limits : StreamingSafetensorsLimits,
) -> Self raise StreamingSafetensorsError {
  guard !root.is_closed() && inspection.shard_count() <= limits.maximum_shards else {
    raise InvalidLimit
  }
  {
    root,
    inspection,
    files: Array::make(inspection.shard_count(), None),
    opened: 0,
    closed: false,
    failed: false,
    busy: false,
  }
}

///|
/// Number of unique shards opened by this session, not by inspection.
pub fn StreamingSafetensorsTransferSession::opened_shards(self : Self) -> Int {
  self.opened
}

///|
pub fn StreamingSafetensorsTransferSession::transfer_ranges(
  self : Self,
  root : @approved_fs.ApprovedRoot,
  requests~ : ArrayView[StreamingSafetensorsTransfer],
  destination_bytes~ : ArrayView[Int64],
  scratch~ : FixedArray[Byte],
  limits~ : StreamingSafetensorsLimits,
  consume~ : (Int, Int64, FixedArray[Byte], Int) -> Bool,
) -> Unit raise StreamingSafetensorsError {
  guard !self.closed && !self.failed && !self.busy && scratch.length() > 0 else {
    raise InvalidLimit
  }
  self.busy = true
  defer {
    self.busy = false
  }
  errdefer {
    self.failed = true
  }
  // Pointer identity is only a cache optimization. Different root wrappers
  // always fall back to the ordinary inspected transfer.
  if !physical_equal(root, self.root) {
    transfer_ranges(
      root,
      self.inspection,
      requests~,
      destination_bytes~,
      scratch~,
      limits~,
      consume~,
    )
    return
  }
  guard !root.is_closed() else { raise FileUnavailable(-1) }
  validate_transfers(self.inspection, requests, destination_bytes, limits)
  for ordinal, shard in self.inspection.shards {
    if !requests.any(request => request.shard_ordinal == ordinal) {
      continue
    }
    let file = match self.files[ordinal] {
      Some(file) => file
      None => {
        let file = root.open_file(shard.locator) catch {
          error => raise map_file_error(ordinal, error)
        }
        // Publish ownership before any fallible read; close drains failures.
        self.files[ordinal] = Some(file)
        guard stamp_file(ordinal, file) == shard.stamp else {
          raise InspectionReplayMismatch(ordinal)
        }
        self.opened += 1
        file
      }
    }
    guard stamp_file(ordinal, file) == shard.stamp else {
      raise FileChanged(ordinal)
    }
    transfer_open_file(ordinal, shard, file, requests, scratch, consume)
    guard stamp_file(ordinal, file) == shard.stamp else {
      raise FileChanged(ordinal)
    }
  }
}

///|
pub fn StreamingSafetensorsTransferSession::close(
  self : Self,
) -> Unit raise StreamingSafetensorsError {
  guard !self.busy else { raise InvalidLimit }
  if self.closed {
    return
  }
  self.failed = true
  let mut first_failure : Int? = None
  for ordinal, file in self.files {
    if file is Some(file) {
      try close_file(ordinal, file) catch {
        _ => if first_failure is None { first_failure = Some(ordinal) }
      } noraise {
        _ => self.files[ordinal] = None
      }
    }
  }
  if first_failure is Some(ordinal) {
    raise FileCloseFailed(ordinal)
  }
  self.closed = true
}
