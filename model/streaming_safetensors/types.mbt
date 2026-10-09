///|
pub(all) enum StreamingSafetensorsDTypeKind {
  BFloat16
  Float32
  OpaqueFixedWidth
} derive(Debug, Eq)

///|
/// Closed caller-registered storage spelling and byte width.
pub struct StreamingSafetensorsDTypeSpec {
  priv kind : StreamingSafetensorsDTypeKind
  priv tag : String
  priv bytes_per_element : Int64
} derive(Debug, Eq)

///|
pub struct StreamingSafetensorsFileDigest {
  priv hex : String
} derive(Debug, Eq, Hash)

///|
/// One strict relative shard and its expected complete-file digest.
pub struct StreamingSafetensorsShardInput {
  priv locator : @approved_fs.ApprovedRelativeLocator
  priv expected_digest : StreamingSafetensorsFileDigest
}

///|
pub struct StreamingSafetensorsLimits {
  priv maximum_shards : Int
  priv maximum_file_bytes : Int64
  priv maximum_header_bytes : Int
  priv maximum_tensors_per_shard : Int
  priv maximum_total_tensors : Int
  priv maximum_rank : Int
  priv maximum_name_bytes : Int
  priv maximum_dtype_specs : Int
  priv authentication_chunk_bytes : Int
  priv maximum_copy_requests : Int
  priv maximum_copy_bytes : Int64
} derive(Debug, Eq)

///|
/// Validated payload-relative range for one tensor.
pub struct StreamingSafetensorsTensorMetadata {
  priv name : String
  priv dtype : StreamingSafetensorsDTypeSpec
  priv shape : ReadOnlyArray[Int64]
  priv data_start : Int64
  priv data_end : Int64
} derive(Debug, Eq)

///|
/// Public authority-free metadata for one authenticated shard.
pub struct StreamingSafetensorsShardMetadata {
  priv file_byte_count : Int64
  priv payload_offset : Int64
  priv payload_byte_count : Int64
  priv file_digest : StreamingSafetensorsFileDigest
  priv tensors : ReadOnlyArray[StreamingSafetensorsTensorMetadata]
} derive(Debug, Eq)

///|
/// Internal retained replay key plus public metadata.
priv struct InspectedShard {
  locator : @approved_fs.ApprovedRelativeLocator
  expected_digest : StreamingSafetensorsFileDigest
  stamp : @approved_fs.ApprovedFileStamp
  metadata : StreamingSafetensorsShardMetadata
}

///|
/// Immutable startup inspection. It owns no open file authority.
pub struct StreamingSafetensorsInspection {
  priv shards : ReadOnlyArray[InspectedShard]
  priv total_tensor_count : Int
}

///|
/// One validated partial or complete tensor copy request.
pub struct StreamingSafetensorsCopyRequest {
  priv shard_ordinal : Int
  priv tensor_ordinal : Int
  priv tensor_byte_offset : Int64
  priv destination_offset : Int
  priv byte_count : Int
} derive(Debug, Eq)

///|
/// Authority-free visit evidence emitted after a direct copy.
pub struct StreamingSafetensorsVisitedRange {
  priv shard_ordinal : Int
  priv tensor_ordinal : Int
  priv tensor_byte_offset : Int64
  priv destination_offset : Int
  priv byte_count : Int
} derive(Debug, Eq)

///|
/// One validated partial or complete tensor copy into a selected final arena.
pub struct StreamingSafetensorsSegmentedCopyRequest {
  priv shard_ordinal : Int
  priv tensor_ordinal : Int
  priv tensor_byte_offset : Int64
  priv destination_arena_ordinal : Int
  priv destination_offset : Int
  priv byte_count : Int
} derive(Debug, Eq)

///|
/// Authority-free evidence emitted after a direct segmented copy.
pub struct StreamingSafetensorsSegmentedVisitedRange {
  priv shard_ordinal : Int
  priv tensor_ordinal : Int
  priv tensor_byte_offset : Int64
  priv destination_arena_ordinal : Int
  priv destination_offset : Int
  priv byte_count : Int
} derive(Debug, Eq)

///|
pub(all) enum StreamingSafetensorsJsonField {
  DType
  Shape
  DataOffsets
} derive(Debug, Eq)

///|
pub(all) suberror StreamingSafetensorsError {
  InvalidLimit
  InvalidDTypeSpec
  DuplicateDTypeTag(Int, Int)
  InvalidDigest
  FileUnavailable(Int)
  UnsupportedFileType(Int)
  FileTruncated(Int)
  FileTooLarge(Int)
  FileChanged(Int)
  FileCloseFailed(Int)
  DigestMismatch(Int)
  MissingHeaderLength(Int)
  HeaderLengthOverflow(Int)
  HeaderTooLarge(Int)
  TruncatedHeader(Int)
  InvalidHeaderEnvelope(Int)
  InvalidHeaderEncoding(Int)
  InvalidHeaderJson(Int)
  DuplicateHeaderKey(shard~ : Int, depth~ : Int, index~ : Int)
  InvalidRoot(Int)
  InvalidDocumentMetadata(Int)
  TooManyTensors(Int)
  InvalidTensor(shard~ : Int, tensor~ : Int)
  MissingTensorField(shard~ : Int, tensor~ : Int, StreamingSafetensorsJsonField)
  UnexpectedTensorField(shard~ : Int, tensor~ : Int)
  InvalidTensorField(shard~ : Int, tensor~ : Int, StreamingSafetensorsJsonField)
  InvalidInteger(shard~ : Int, tensor~ : Int, StreamingSafetensorsJsonField)
  InvalidTensorName(shard~ : Int, tensor~ : Int)
  UnknownDType(shard~ : Int, tensor~ : Int)
  InvalidRank(shard~ : Int, tensor~ : Int)
  InvalidDimension(shard~ : Int, tensor~ : Int)
  TensorSizeOverflow(shard~ : Int, tensor~ : Int)
  InvalidByteRange(shard~ : Int, tensor~ : Int)
  RangeOutsidePayload(shard~ : Int, tensor~ : Int)
  RangeSizeMismatch(shard~ : Int, tensor~ : Int)
  DuplicateTensorName(
    first_shard~ : Int,
    first_tensor~ : Int,
    second_shard~ : Int,
    second_tensor~ : Int
  )
  OverlappingTensorRanges(shard~ : Int, first~ : Int, second~ : Int)
  InvalidCopyRequest(Int)
  CopyRangeOutsideTensor(Int)
  CopyRangeOutsideDestination(Int)
  InvalidDestinationArena(Int)
  OverlappingDestinationRanges(Int, Int)
  CopyBudgetExceeded
  InspectionReplayMismatch(Int)
  VisitorRejected(Int)
} derive(Debug, Eq)
