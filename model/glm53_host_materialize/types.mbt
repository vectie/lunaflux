///|
pub(all) enum Glm53HostLimit {
  HeaderBytes
  TensorCount
  TensorRank
  TensorNameBytes
  SourceCount
  FileBytes
  ArenaBytes
} derive(Debug, Eq)

///|
pub(all) enum Glm53HostDigestKind {
  ExecutionIdentity
  ManifestLayout
  NumericSchema
  NumericBinding
} derive(Debug, Eq)

///|
pub(all) suberror Glm53HostMaterializeError {
  InvalidLimit(Glm53HostLimit)
  UnsupportedPhysicalEncoding
  DigestMismatch(Glm53HostDigestKind)
  ProfileMismatch
  SourceCountMismatch
  StreamingAdmission(@streaming_safetensors.StreamingSafetensorsError)
  PayloadCoverageMismatch(shard~ : Int)
  TensorCountMismatch(expected~ : Int, actual~ : Int)
  TensorNameMismatch(ordinal~ : Int)
  TensorDTypeMismatch(ordinal~ : Int)
  TensorShapeMismatch(ordinal~ : Int)
  TensorByteCountMismatch(ordinal~ : Int)
  NumericTensorMismatch(ordinal~ : Int)
  ArenaSizeOverflow
  Closed
  TensorOrdinalOutOfRange
  TensorByteOutOfRange
} derive(Debug, Eq)

///|
pub struct Glm53HostShardSource {
  priv locator : @approved_fs.ApprovedRelativeLocator
  priv artifact_digest : @streaming_safetensors.StreamingSafetensorsFileDigest
}

///|
pub fn Glm53HostShardSource::new(
  locator~ : @approved_fs.ApprovedRelativeLocator,
  artifact_digest~ : @streaming_safetensors.StreamingSafetensorsFileDigest,
) -> Glm53HostShardSource {
  { locator, artifact_digest, }
}

///|
pub struct Glm53HostExpectedDigests {
  priv execution_identity : @spec.ModelIdentity
  priv manifest_layout : @spec.PlanDigest
  priv numeric_schema : @numeric_contract.ModelNumericSchemaDigest
  priv numeric_binding : @spec.PlanDigest
}

///|
pub fn Glm53HostExpectedDigests::new(
  execution_identity~ : @spec.ModelIdentity,
  manifest_layout~ : @spec.PlanDigest,
  numeric_schema~ : @numeric_contract.ModelNumericSchemaDigest,
  numeric_binding~ : @spec.PlanDigest,
) -> Glm53HostExpectedDigests {
  { execution_identity, manifest_layout, numeric_schema, numeric_binding, }
}

///|
/// Canonical identity of the authenticated shard set and exact final host
/// segmentation. It is derived internally and has no public constructor.
pub struct Glm53HostSourceDigest {
  priv hex : String
} derive(Debug, Eq, Hash)

///|
pub struct Glm53HostMaterializationLimits {
  priv max_header_bytes : Int
  priv max_tensor_count : Int
  priv max_rank : Int
  priv max_name_bytes : Int
  priv max_source_count : Int
  priv max_file_bytes : Int64
  priv max_arena_bytes : Int
  priv max_total_arena_bytes : Int64
}

///|
pub fn Glm53HostMaterializationLimits::new(
  max_header_bytes~ : Int,
  max_tensor_count~ : Int,
  max_rank~ : Int,
  max_name_bytes~ : Int,
  max_source_count~ : Int,
  max_file_bytes~ : Int64,
  max_arena_bytes~ : Int,
  max_total_arena_bytes~ : Int64,
) -> Glm53HostMaterializationLimits raise Glm53HostMaterializeError {
  if max_header_bytes <= 0 || max_header_bytes > 64 * 1024 * 1024 {
    raise InvalidLimit(HeaderBytes)
  }
  if max_tensor_count <= 0 || max_tensor_count > 100000 {
    raise InvalidLimit(TensorCount)
  }
  if max_rank <= 0 || max_rank > 16 {
    raise InvalidLimit(TensorRank)
  }
  if max_name_bytes <= 0 || max_name_bytes > 4096 {
    raise InvalidLimit(TensorNameBytes)
  }
  if max_source_count <= 0 || max_source_count > 4096 {
    raise InvalidLimit(SourceCount)
  }
  if max_file_bytes <= 8L {
    raise InvalidLimit(FileBytes)
  }
  if max_arena_bytes <= 0 || max_arena_bytes > 1024 * 1024 * 1024 {
    raise InvalidLimit(ArenaBytes)
  }
  if max_total_arena_bytes <= 0L {
    raise InvalidLimit(ArenaBytes)
  }
  {
    max_header_bytes,
    max_tensor_count,
    max_rank,
    max_name_bytes,
    max_source_count,
    max_file_bytes,
    max_arena_bytes,
    max_total_arena_bytes,
  }
}

///|
pub struct Glm53HostTensorSpan {
  priv arena_index : Int
  priv offset_bytes : Int
  priv byte_length : Int
}

///|
pub fn Glm53HostTensorSpan::arena_index(self : Self) -> Int {
  self.arena_index
}

///|
pub fn Glm53HostTensorSpan::offset_bytes(self : Self) -> Int {
  self.offset_bytes
}

///|
pub fn Glm53HostTensorSpan::byte_length(self : Self) -> Int {
  self.byte_length
}

///|
pub struct Glm53HostTensorRegion {
  priv ordinal : Int
  priv byte_length : Int64
  priv spans : ReadOnlyArray[Glm53HostTensorSpan]
}

///|
pub fn Glm53HostTensorRegion::ordinal(self : Self) -> Int {
  self.ordinal
}

///|
pub fn Glm53HostTensorRegion::byte_length(self : Self) -> Int64 {
  self.byte_length
}

///|
pub fn Glm53HostTensorRegion::span_count(self : Self) -> Int {
  self.spans.length()
}

///|
pub fn Glm53HostTensorRegion::span(
  self : Self,
  index : Int,
) -> Glm53HostTensorSpan? {
  self.spans.get(index)
}

///|
priv struct Glm53HostArena {
  mut bytes : FixedArray[Byte]
}

///|
pub struct Glm53HostWeights {
  priv execution_identity : @spec.ModelIdentity
  priv manifest_layout : @spec.PlanDigest
  priv numeric_schema : @numeric_contract.ModelNumericSchemaDigest
  priv numeric_binding : @spec.PlanDigest
  priv source_digest : Glm53HostSourceDigest
  priv regions : ReadOnlyArray[Glm53HostTensorRegion]
  priv arenas : Array[Glm53HostArena]
  priv total_bytes : Int64
  priv mut open : Bool
}
