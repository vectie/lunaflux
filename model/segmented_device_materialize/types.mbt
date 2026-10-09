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
  priv maximum_source_arenas : Int
  priv maximum_regions : Int
  priv maximum_host_arena_bytes : Int
  priv maximum_total_source_bytes : Int64
  priv maximum_device_arena_bytes : Int64
}

///|
pub struct SegmentedDeviceTensorRegionInput {
  priv tensor_ordinal : Int
  priv source_arena_ordinal : Int
  priv source_offset : Int
  priv byte_count : Int
} derive(Debug, Eq)

///|
pub struct SegmentedDeviceTensorRegion {
  priv tensor_ordinal : Int
  priv source_arena_ordinal : Int
  priv source_offset : Int
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
  priv source_arena_lengths : ReadOnlyArray[Int]
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
  InvalidSourceArena(Int)
  SourceByteBudgetExceeded
  InvalidTensorOrdinal(Int)
  InvalidSourceRegion(Int)
  OverlappingSourceRegions(Int, Int)
  DeviceLayoutOverflow(Int)
  DeviceArenaTooLarge
  IdentityMismatch
  SourceDigestMismatch
  LayoutDigestMismatch
  SourceArenaShapeMismatch(Int)
  HostArenaBorrowClosed(Int)
  HostArenaAlreadyProvided(Int)
  HostArenaLengthMismatch(Int)
  HostArenaNotProvided(Int)
  HostArenaConsumerRejected(Int)
  HostSourceUnavailable(Int)
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
