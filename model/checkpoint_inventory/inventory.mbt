///|
pub(all) suberror CheckpointInventoryError {
  InvalidLimits
  InvalidEntry(Int)
  TooManyShards
  EmptyInventory
  DuplicateLocator(Int)
}

///|
/// Startup-only inventory. Entries contain a declared label, two spaces, and
/// an approved relative safetensors locator. No shell syntax is interpreted.
/// Plain sha256sum records remain readable as unverified labels. `label:HEX`
/// explicitly denotes a supplied checkpoint identity, not a shard checksum.
/// Neither form triggers payload hashing during inspection or loading.
pub struct CheckpointInventory {
  priv shards : Array[@streaming.StreamingSafetensorsShardInput]
}

///|
pub fn CheckpointInventory::parse(
  document : StringView,
  maximum_bytes~ : Int,
  maximum_shards~ : Int,
) -> Self raise {
  guard maximum_bytes > 0 && maximum_shards > 0 else {
    raise CheckpointInventoryError::InvalidLimits
  }
  guard document.length() <= maximum_bytes else {
    raise CheckpointInventoryError::InvalidLimits
  }
  guard @utf8.encode(document.to_owned()).length() <= maximum_bytes else {
    raise CheckpointInventoryError::InvalidLimits
  }
  let shards : Array[@streaming.StreamingSafetensorsShardInput] = []
  let names : Map[String, Unit] = Map([])
  for ordinal, line in document.split("\n") {
    if line.is_empty() {
      continue
    }
    let record = if line.has_prefix("label:") { line[6:] } else { line }
    guard record.length() > 66 && record[64:66] == "  " else {
      raise CheckpointInventoryError::InvalidEntry(ordinal)
    }
    guard shards.length() < maximum_shards else {
      raise CheckpointInventoryError::TooManyShards
    }
    let name = record[66:].to_owned()
    guard name.has_suffix(".safetensors") else {
      raise CheckpointInventoryError::InvalidEntry(ordinal)
    }
    guard !names.contains(name) else {
      raise CheckpointInventoryError::DuplicateLocator(ordinal)
    }
    let locator = @approved_fs.ApprovedRelativeLocator::new(name)
    let digest = @streaming.StreamingSafetensorsFileDigest::from_sha256(
      record[:64].to_owned(),
    )
    names[name] = ()
    shards.push(
      @streaming.StreamingSafetensorsShardInput::new(
        locator~,
        expected_digest=digest,
      ),
    )
  }
  guard !shards.is_empty() else {
    raise CheckpointInventoryError::EmptyInventory
  }
  { shards, }
}

///|
pub fn CheckpointInventory::shards(
  self : Self,
) -> ArrayView[@streaming.StreamingSafetensorsShardInput] {
  self.shards[:]
}
