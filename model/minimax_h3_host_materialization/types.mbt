///|
/// Startup-only failures. Names and shard bytes never enter error payloads.
pub(all) suberror MiniMaxH3HostMaterializationError {
  InvalidLimit
  IdentityMismatch
  ManifestDigestMismatch
  InvalidShardCount
  UnreferencedShard(Int32)
  InvalidIndexCount(expected~ : Int32, actual~ : Int32)
  MissingIndex(Int32)
  DuplicateIndex(Int32)
  UnknownIndex(Int32)
  ShardIndexOutsideManifest(Int32)
  InvalidShardRange(Int32)
  InvalidOfficialShape(Int32)
  InvalidDType(Int32)
  TensorTooLarge(Int32)
  HostBudgetExceeded
  ArenaCountExceeded
  ShardCountMismatch
  ShardPayloadMismatch(Int32)
  StreamingFailure
  StreamingShardMismatch(Int32)
  StreamingTensorCountMismatch(Int32)
  StreamingTensorMismatch(Int32)
  ArenaReleased
  TensorIndexOutsideArena(Int32)
  TensorByteOutsideArena(Int32)
} derive(Debug, Eq)

///|
/// Bounded startup limits. Arenas are intentionally smaller than MoonBit's
/// signed-Int allocation ceiling and components may span multiple arenas.
pub struct MiniMaxH3HostMaterializationLimits {
  priv maximum_shards : Int32
  priv maximum_tensors : Int32
  priv maximum_header_bytes : Int32
  priv maximum_payload_bytes_per_shard : Int64
  priv maximum_total_host_bytes : Int64
  priv maximum_arena_bytes : Int64
  priv maximum_arenas : Int32
  priv maximum_name_bytes : Int32
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
  priv shard_index : Int32
} derive(Debug, Eq)

///|
/// Supplied model-plan association label, not a component/layout checksum.
/// Components and arena placements may share it; their typed metadata is kept
/// separately in the manifest.
pub struct MiniMaxH3HostManifestDigest {
  priv hex : String
} derive(Debug, Eq)

///|
priv struct MiniMaxH3HostManifestTensor {
  binding_index : Int32
  role : @minimax_h3_weights.MiniMaxH3TensorRole
  tensor : @safetensors.TensorMetadata
  shard_index : Int32
  arena_index : Int32
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
  priv arena_index : Int32
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
