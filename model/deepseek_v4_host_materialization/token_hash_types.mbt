///|
pub(all) suberror DeepSeekV4TokenHashMaterializationError {
  TokenHashInvalidLimit
  TokenHashSourceReleased
  TokenHashIdentityMismatch
  TokenHashTableCountMismatch(expected~ : Int32, actual~ : Int32)
  TokenHashInvalidBinding(Int32)
  TokenHashMissingLayer(Int32)
  TokenHashDuplicateLayer(Int32)
  TokenHashOutputBudgetExceeded
  TokenHashTableIndexOutOfBounds(Int32)
  TokenHashNarrowingFailure(
    layer~ : Int32,
    @checked_integer_narrow.CheckedIntegerNarrowError
  )
} derive(Debug, Eq)

///|
pub struct DeepSeekV4TokenHashMaterializationLimits {
  priv maximum_tables : Int32
  priv maximum_total_i32_bytes : Int64
}

///|
priv struct DeepSeekV4TokenHashTableEntry {
  layer_index : Int32
  source_tensor_index : Int32
  table : @checked_integer_narrow.CheckedI32Table
}

///|
/// Explicitly releasable derived I32 token-hash tables, ordered by decoder
/// layer and bound to the exact source model/manifest identity.
pub struct DeepSeekV4TokenHashTables {
  priv identity : @spec.ModelIdentity
  priv source_manifest_digest : DeepSeekV4HostManifestDigest
  priv entries : ReadOnlyArray[DeepSeekV4TokenHashTableEntry]
  priv mut released : Bool
}
