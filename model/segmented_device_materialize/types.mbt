///|
pub struct SegmentedHostSourceDigest {
  priv hex : String
} derive(Debug, Eq, Hash)

///|
pub struct SegmentedDeviceLayoutDigest {
  priv hex : String
} derive(Debug, Eq, Hash)

///|
pub struct SegmentedDeviceMaterializationLimits {
  priv alignment_bytes : Int64
  priv maximum_source_arenas : Int32
  priv maximum_regions : Int32
  priv maximum_host_arena_bytes : Int32
  priv maximum_total_source_bytes : Int64
  priv maximum_device_arena_bytes : Int64
}

///|
pub struct SegmentedDeviceTensorRegionInput {
  priv tensor_ordinal : Int32
  priv source_arena_ordinal : Int32
  priv source_offset : Int32
  priv byte_count : Int32
} derive(Debug, Eq)

///|
pub struct SegmentedDeviceTensorRegion {
  priv tensor_ordinal : Int32
  priv source_arena_ordinal : Int32
  priv source_offset : Int32
  priv device_offset : Int64
  priv byte_count : Int64
} derive(Debug, Eq)

///|
pub struct SegmentedDeviceLayout {
  priv source_identity : @spec.ModelIdentity
  priv source_digest : SegmentedHostSourceDigest
  priv digest : SegmentedDeviceLayoutDigest
  priv alignment_bytes : Int64
  priv device_arena_bytes : Int64
  priv materialized_bytes : Int64
  priv source_arena_lengths : ReadOnlyArray[Int32]
  priv regions : ReadOnlyArray[SegmentedDeviceTensorRegion]
} derive(Debug, Eq)

///|
pub struct SegmentedDeviceWeights {
  priv allocation : @device.Allocation
  priv layout : SegmentedDeviceLayout
}

///|
pub(all) enum SegmentedDevicePreparation {
  SegmentedDeviceWeightsReady(SegmentedDeviceWeights)
  SegmentedDeviceCleanupRequired(FailedSegmentedDevicePreparation)
}

///|
pub(all) enum SegmentedDeviceFailureStage {
  AllocationRegionValidation
  Transfer
} derive(Debug, Eq)

///|
pub struct FailedSegmentedDevicePreparation {
  priv allocation : @device.Allocation
  priv cause : SegmentedDeviceMaterializationError
  priv stage : SegmentedDeviceFailureStage
  priv mut cleanup_failure : @device.DeviceError
  priv mut open : Bool
}

///|
pub(all) suberror SegmentedDeviceMaterializationError {
  InvalidDigest
  InvalidLimit
  EmptySourceArenas
  TooManySourceArenas
  EmptyRegions
  TooManyRegions
  InvalidSourceArena(Int32)
  SourceByteBudgetExceeded
  InvalidTensorOrdinal(Int32)
  InvalidSourceRegion(Int32)
  OverlappingSourceRegions(Int32, Int32)
  DeviceLayoutOverflow(Int32)
  DeviceArenaTooLarge
  IdentityMismatch
  SourceDigestMismatch
  LayoutDigestMismatch
  SourceArenaShapeMismatch(Int32)
  HostArenaBorrowClosed(Int32)
  HostArenaAlreadyProvided(Int32)
  HostArenaLengthMismatch(Int32)
  HostArenaNotProvided(Int32)
  HostArenaConsumerRejected(Int32)
  HostSourceUnavailable(Int32)
  DeviceFailed(@device.DeviceError)
  CleanupFailed(stage~ : SegmentedDeviceFailureStage, @device.DeviceError)
} derive(Debug, Eq)

///|
priv struct PreparedSegmentedResource[T] {
  resource : T
  layout : SegmentedDeviceLayout
}

///|
priv enum SegmentedResourcePreparation[T] {
  SegmentedResourceReady(PreparedSegmentedResource[T])
  SegmentedResourceCleanupRequired(FailedSegmentedResourcePreparation[T])
}

///|
priv struct FailedSegmentedResourcePreparation[T] {
  resource : T
  cause : SegmentedDeviceMaterializationError
  stage : SegmentedDeviceFailureStage
  cleanup_failure : @device.DeviceError
}

///|
pub extend SegmentedHostSourceDigest with @debug.Debug::{to_repr}

///|
pub extend SegmentedHostSourceDigest with Eq::{not_equal, equal}

///|
pub extend SegmentedHostSourceDigest with Hash::{hash, hash_combine}

///|
pub extend SegmentedDeviceLayoutDigest with @debug.Debug::{to_repr}

///|
pub extend SegmentedDeviceLayoutDigest with Eq::{not_equal, equal}

///|
pub extend SegmentedDeviceLayoutDigest with Hash::{hash, hash_combine}

///|
pub extend SegmentedDeviceTensorRegionInput with @debug.Debug::{to_repr}

///|
pub extend SegmentedDeviceTensorRegionInput with Eq::{not_equal, equal}

///|
pub extend SegmentedDeviceTensorRegion with @debug.Debug::{to_repr}

///|
pub extend SegmentedDeviceTensorRegion with Eq::{not_equal, equal}

///|
pub extend SegmentedDeviceLayout with @debug.Debug::{to_repr}

///|
pub extend SegmentedDeviceLayout with Eq::{not_equal, equal}

///|
pub extend SegmentedDeviceFailureStage with @debug.Debug::{to_repr}

///|
pub extend SegmentedDeviceFailureStage with Eq::{not_equal, equal}

///|
pub extend SegmentedDeviceMaterializationError with @debug.Debug::{to_repr}

///|
pub extend SegmentedDeviceMaterializationError with Eq::{not_equal, equal}
