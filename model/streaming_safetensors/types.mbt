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
/// One relative shard and its declared inventory label. Loading does not hash it.
pub struct StreamingSafetensorsShardInput {
  priv locator : @approved_fs.ApprovedRelativeLocator
  priv expected_digest : StreamingSafetensorsFileDigest
}

///|
pub struct StreamingSafetensorsLimits {
  priv maximum_shards : Int32
  priv maximum_file_bytes : Int64
  priv maximum_header_bytes : Int32
  priv maximum_tensors_per_shard : Int32
  priv maximum_total_tensors : Int32
  priv maximum_rank : Int32
  priv maximum_name_bytes : Int32
  priv maximum_dtype_specs : Int32
  priv maximum_copy_requests : Int32
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
/// Public metadata for one structurally inspected shard, not hash-verified bytes.
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
  stamp : @approved_fs.ApprovedFileStamp
  metadata : StreamingSafetensorsShardMetadata
}

///|
/// Immutable startup inspection. It owns no open file authority.
pub struct StreamingSafetensorsInspection {
  priv shards : ReadOnlyArray[InspectedShard]
  priv total_tensor_count : Int32
}

///|
/// One validated partial or complete tensor copy request.
pub struct StreamingSafetensorsCopyRequest {
  priv shard_ordinal : Int32
  priv tensor_ordinal : Int32
  priv tensor_byte_offset : Int64
  priv destination_offset : Int32
  priv byte_count : Int32
} derive(Debug, Eq)

///|
/// Authority-free visit evidence emitted after a direct copy.
pub struct StreamingSafetensorsVisitedRange {
  priv shard_ordinal : Int32
  priv tensor_ordinal : Int32
  priv tensor_byte_offset : Int64
  priv destination_offset : Int32
  priv byte_count : Int32
} derive(Debug, Eq)

///|
/// One validated partial or complete tensor copy into a selected final arena.
pub struct StreamingSafetensorsSegmentedCopyRequest {
  priv shard_ordinal : Int32
  priv tensor_ordinal : Int32
  priv tensor_byte_offset : Int64
  priv destination_arena_ordinal : Int32
  priv destination_offset : Int32
  priv byte_count : Int32
} derive(Debug, Eq)

///|
/// Authority-free evidence emitted after a direct segmented copy.
pub struct StreamingSafetensorsSegmentedVisitedRange {
  priv shard_ordinal : Int32
  priv tensor_ordinal : Int32
  priv tensor_byte_offset : Int64
  priv destination_arena_ordinal : Int32
  priv destination_offset : Int32
  priv byte_count : Int32
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
  DuplicateDTypeTag(Int32, Int32)
  InvalidDigest
  FileUnavailable(Int32)
  UnsupportedFileType(Int32)
  FileTruncated(Int32)
  FileTooLarge(Int32)
  FileChanged(Int32)
  FileCloseFailed(Int32)
  MissingHeaderLength(Int32)
  HeaderLengthOverflow(Int32)
  HeaderTooLarge(Int32)
  TruncatedHeader(Int32)
  InvalidHeaderEnvelope(Int32)
  InvalidHeaderEncoding(Int32)
  InvalidHeaderJson(Int32)
  DuplicateHeaderKey(shard~ : Int32, depth~ : Int32, index~ : Int32)
  InvalidRoot(Int32)
  InvalidDocumentMetadata(Int32)
  TooManyTensors(Int32)
  InvalidTensor(shard~ : Int32, tensor~ : Int32)
  MissingTensorField(
    shard~ : Int32,
    tensor~ : Int32,
    StreamingSafetensorsJsonField
  )
  UnexpectedTensorField(shard~ : Int32, tensor~ : Int32)
  InvalidTensorField(
    shard~ : Int32,
    tensor~ : Int32,
    StreamingSafetensorsJsonField
  )
  InvalidInteger(shard~ : Int32, tensor~ : Int32, StreamingSafetensorsJsonField)
  InvalidTensorName(shard~ : Int32, tensor~ : Int32)
  UnknownDType(shard~ : Int32, tensor~ : Int32)
  InvalidRank(shard~ : Int32, tensor~ : Int32)
  InvalidDimension(shard~ : Int32, tensor~ : Int32)
  TensorSizeOverflow(shard~ : Int32, tensor~ : Int32)
  InvalidByteRange(shard~ : Int32, tensor~ : Int32)
  RangeOutsidePayload(shard~ : Int32, tensor~ : Int32)
  RangeSizeMismatch(shard~ : Int32, tensor~ : Int32)
  DuplicateTensorName(
    first_shard~ : Int32,
    first_tensor~ : Int32,
    second_shard~ : Int32,
    second_tensor~ : Int32
  )
  OverlappingTensorRanges(shard~ : Int32, first~ : Int32, second~ : Int32)
  InvalidCopyRequest(Int32)
  CopyRangeOutsideTensor(Int32)
  CopyRangeOutsideDestination(Int32)
  InvalidDestinationArena(Int32)
  OverlappingDestinationRanges(Int32, Int32)
  CopyBudgetExceeded
  InspectionReplayMismatch(Int32)
  VisitorRejected(Int32)
} derive(Debug, Eq)
