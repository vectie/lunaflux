///|
pub fn DeepSeekV4TokenHashTables::identity(self : Self) -> @spec.ModelIdentity {
  self.identity
}

///|
pub fn DeepSeekV4TokenHashTables::source_manifest_digest(
  self : Self,
) -> DeepSeekV4HostManifestDigest {
  self.source_manifest_digest
}

///|
pub fn DeepSeekV4TokenHashTables::table_count(self : Self) -> Int {
  self.entries.length()
}

///|
fn DeepSeekV4TokenHashTables::entry(
  self : Self,
  index : Int,
) -> DeepSeekV4TokenHashTableEntry raise DeepSeekV4TokenHashMaterializationError {
  if self.released {
    raise TokenHashSourceReleased
  }
  match self.entries.get(index) {
    Some(entry) => entry
    None => raise TokenHashTableIndexOutOfBounds(index)
  }
}

///|
pub fn DeepSeekV4TokenHashTables::layer_index(
  self : Self,
  index : Int,
) -> Int raise DeepSeekV4TokenHashMaterializationError {
  self.entry(index).layer_index
}

///|
pub fn DeepSeekV4TokenHashTables::source_tensor_index(
  self : Self,
  index : Int,
) -> Int raise DeepSeekV4TokenHashMaterializationError {
  self.entry(index).source_tensor_index
}

///|
pub fn DeepSeekV4TokenHashTables::table_byte_count(
  self : Self,
  index : Int,
) -> Int raise DeepSeekV4TokenHashMaterializationError {
  self.entry(index).table.byte_count()
}

///|
pub fn DeepSeekV4TokenHashTables::table_digest(
  self : Self,
  index : Int,
) -> @checked_integer_narrow.CheckedI32TableDigest raise DeepSeekV4TokenHashMaterializationError {
  self.entry(index).table.digest()
}

///|
pub fn DeepSeekV4TokenHashTables::provide_table_arena(
  self : Self,
  index : Int,
  consumer : @segmented_host_borrow.SegmentedHostArenaConsumer,
) -> Unit raise @segmented_host_borrow.SegmentedHostBorrowError {
  if self.released {
    raise SourceUnavailable(index)
  }
  guard self.entries.get(index) is Some(entry) else {
    raise SourceUnavailable(index)
  }
  entry.table.provide_arena(consumer)
}

///|
pub fn DeepSeekV4TokenHashTables::is_released(self : Self) -> Bool {
  self.released
}

///|
/// Zeroes and releases every derived table exactly once.
pub fn DeepSeekV4TokenHashTables::release(self : Self) -> Bool {
  if self.released {
    return false
  }
  for entry in self.entries {
    ignore(entry.table.release())
  }
  self.released = true
  true
}
