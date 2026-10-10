///|
/// Startup-only failures. Names and shard bytes never enter error payloads.
pub(all) suberror MiniMaxH3HostMaterializationError {
  InvalidLimit
  IdentityMismatch
  ManifestDigestMismatch
  InvalidShardCount
  UnreferencedShard(Int)
  InvalidIndexCount(expected~ : Int, actual~ : Int)
  MissingIndex(Int)
  DuplicateIndex(Int)
  UnknownIndex(Int)
  ShardIndexOutsideManifest(Int)
  InvalidShardRange(Int)
  InvalidOfficialShape(Int)
  InvalidDType(Int)
  TensorTooLarge(Int)
  HostBudgetExceeded
  ArenaCountExceeded
  ShardCountMismatch
  ShardPayloadMismatch(Int)
  StreamingFailure
  StreamingShardMismatch(Int)
  StreamingTensorCountMismatch(Int)
  StreamingTensorMismatch(Int)
  ArenaReleased
  TensorIndexOutsideArena(Int)
  TensorByteOutsideArena(Int)
} derive(Debug, Eq)

///|
/// Bounded startup limits. Arenas are intentionally smaller than MoonBit's
/// signed-Int allocation ceiling and components may span multiple arenas.
pub struct MiniMaxH3HostMaterializationLimits {
  priv maximum_shards : Int
  priv maximum_tensors : Int
  priv maximum_header_bytes : Int
  priv maximum_payload_bytes_per_shard : Int64
  priv maximum_total_host_bytes : Int64
  priv maximum_arena_bytes : Int64
  priv maximum_arenas : Int
  priv maximum_name_bytes : Int
}

///|
/// One canonical shard declaration. Array position is the stable shard index.
pub struct MiniMaxH3ShardDeclarationInput {
  priv file_digest : @spec.ContentDigest
  priv payload_bytes : Int64
} derive(Debug, Eq)

///|
/// One normalized safetensors index membership claim.
pub struct MiniMaxH3TensorIndexInput {
  priv tensor_name : String
  priv shard_index : Int
} derive(Debug, Eq)

///|
/// Digest of identity, component, shard declarations, exact tensor metadata,
/// membership, and final arena layout.
pub struct MiniMaxH3HostManifestDigest {
  priv hex : String
} derive(Debug, Eq)

///|
priv struct MiniMaxH3HostManifestTensor {
  binding_index : Int
  role : @minimax_h3_weights.MiniMaxH3TensorRole
  tensor : @safetensors.TensorMetadata
  shard_index : Int
  arena_index : Int
  arena_offset : Int64
}

///|
/// Immutable exact component manifest. Both VAE components are admitted only
/// by their exact official schemas.
pub struct MiniMaxH3HostComponentManifest {
  priv component : @minimax_h3_weights.MiniMaxH3WeightComponent
  priv identity : @spec.ModelIdentity
  priv digest : MiniMaxH3HostManifestDigest
  priv shards : ReadOnlyArray[MiniMaxH3ShardDeclarationInput]
  priv tensors : ReadOnlyArray[MiniMaxH3HostManifestTensor]
  priv arena_sizes : ReadOnlyArray[Int64]
  priv total_tensor_bytes : Int64
}

///|
/// One approved-root-relative source. Its array position must equal the
/// canonical shard index in the component manifest.
pub struct MiniMaxH3ShardSource {
  priv locator : @approved_fs.ApprovedRelativeLocator
}

///|
/// Metadata for one tensor copied directly into its final bounded host arena.
pub struct MiniMaxH3HostTensor {
  priv name : String
  priv role : @minimax_h3_weights.MiniMaxH3TensorRole
  priv shape : ReadOnlyArray[Int64]
  priv arena_index : Int
  priv arena_offset : Int64
  priv byte_length : Int64
} derive(Debug, Eq)

///|
/// Explicitly releasable, component-scoped mixed-precision host arenas. This
/// type is not a complete MiniMax H3 model. It may own one exact VAE component,
/// but never claims VAE execution or whole-model readiness.
pub struct MiniMaxH3HostComponentArena {
  priv component : @minimax_h3_weights.MiniMaxH3WeightComponent
  priv identity : @spec.ModelIdentity
  priv manifest_digest : MiniMaxH3HostManifestDigest
  priv tensors : ReadOnlyArray[MiniMaxH3HostTensor]
  priv arenas : Array[FixedArray[Byte]?]
  priv total_tensor_bytes : Int64
  priv mut released : Bool
}
