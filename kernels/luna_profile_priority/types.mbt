///|
const LUNA_KERNEL_PROFILE_MAX_NANOS : Int64 = 9_000_000_000_000L

///|
fn luna_kernel_profile_is_sha256(value : StringView) -> Bool {
  if value.length() != 64 {
    return false
  }
  for unit in value {
    if !(unit is ('0'..='9' | 'a'..='f')) {
      return false
    }
  }
  true
}

///|
pub struct LunaKernelProfileDigest(String) derive(Debug, Eq, Hash)

///|
pub fn LunaKernelProfileDigest::from_sha256(
  value : String,
) -> LunaKernelProfileDigest raise LunaKernelProfileError {
  if !luna_kernel_profile_is_sha256(value) {
    raise LunaKernelProfileInvalid(Digest)
  }
  LunaKernelProfileDigest(value)
}

///|
pub fn LunaKernelProfileDigest::as_hex(
  self : LunaKernelProfileDigest,
) -> String {
  self.0
}

///|
pub struct LunaKernelProfileLimits {
  priv maximum_observations : Int
  priv maximum_canonical_bytes : Int
} derive(Debug, Eq)

///|
pub fn LunaKernelProfileLimits::new(
  maximum_observations~ : Int,
  maximum_canonical_bytes~ : Int,
) -> LunaKernelProfileLimits raise LunaKernelProfileError {
  if maximum_observations <= 0 ||
    maximum_observations > 8192 ||
    maximum_canonical_bytes <= 0 ||
    maximum_canonical_bytes > 16_777_216 {
    raise LunaKernelProfileInvalid(Limit)
  }
  { maximum_observations, maximum_canonical_bytes }
}

///|
pub struct LunaKernelProfileObservation {
  priv operation_id : @plan.OperationId
  priv shape : @device_profile.ExactExecutionShape
  priv call_count : Int64
  priv self_time_ns : Int64
  priv inclusive_time_ns : Int64
} derive(Debug, Eq)

///|
pub fn LunaKernelProfileObservation::new(
  operation_id~ : @plan.OperationId,
  shape~ : @device_profile.ExactExecutionShape,
  call_count~ : Int64,
  self_time_ns~ : Int64,
  inclusive_time_ns~ : Int64,
) -> LunaKernelProfileObservation raise LunaKernelProfileError {
  if call_count <= 0L ||
    call_count > 1_000_000_000L ||
    self_time_ns <= 0L ||
    self_time_ns > LUNA_KERNEL_PROFILE_MAX_NANOS ||
    inclusive_time_ns < self_time_ns ||
    inclusive_time_ns > LUNA_KERNEL_PROFILE_MAX_NANOS {
    raise LunaKernelProfileInvalid(Timing)
  }
  { operation_id, shape, call_count, self_time_ns, inclusive_time_ns }
}

///|
pub fn LunaKernelProfileObservation::operation_id(
  self : LunaKernelProfileObservation,
) -> @plan.OperationId {
  self.operation_id
}

///|
pub fn LunaKernelProfileObservation::shape(
  self : LunaKernelProfileObservation,
) -> @device_profile.ExactExecutionShape {
  self.shape
}

///|
pub fn LunaKernelProfileObservation::call_count(
  self : LunaKernelProfileObservation,
) -> Int64 {
  self.call_count
}

///|
pub fn LunaKernelProfileObservation::self_time_ns(
  self : LunaKernelProfileObservation,
) -> Int64 {
  self.self_time_ns
}

///|
pub fn LunaKernelProfileObservation::inclusive_time_ns(
  self : LunaKernelProfileObservation,
) -> Int64 {
  self.inclusive_time_ns
}

///|
pub struct LunaProfiledOperationPriority {
  priv operation_id : @plan.OperationId
  priv operation_kind : @plan.OperationKind
  priv capability : @plan.KernelCapabilityId
  priv attributed_self_time_ns : Int64
  priv attributed_share_ppm : Int
} derive(Debug, Eq)

///|
pub fn LunaProfiledOperationPriority::operation_id(
  self : LunaProfiledOperationPriority,
) -> @plan.OperationId {
  self.operation_id
}

///|
pub fn LunaProfiledOperationPriority::operation_kind(
  self : LunaProfiledOperationPriority,
) -> @plan.OperationKind {
  self.operation_kind
}

///|
pub fn LunaProfiledOperationPriority::capability(
  self : LunaProfiledOperationPriority,
) -> @plan.KernelCapabilityId {
  self.capability
}

///|
pub fn LunaProfiledOperationPriority::attributed_self_time_ns(
  self : LunaProfiledOperationPriority,
) -> Int64 {
  self.attributed_self_time_ns
}

///|
pub fn LunaProfiledOperationPriority::attributed_share_ppm(
  self : LunaProfiledOperationPriority,
) -> Int {
  self.attributed_share_ppm
}

///|
pub struct LunaKernelProfileCapture {
  priv identity : @spec.ModelIdentity
  priv target : @catalog.DeviceTarget
  priv workload : LunaKernelProfileDigest
  priv profiler : LunaKernelProfileDigest
  priv sampled_wall_time_ns : Int64
  priv attributed_self_time_ns : Int64
  priv observations : ReadOnlyArray[LunaKernelProfileObservation]
  priv priority : LunaProfiledOperationPriority
  priv canonical : Bytes
  priv digest : LunaKernelProfileDigest
} derive(Debug, Eq)

///|
pub fn LunaKernelProfileCapture::identity(
  self : LunaKernelProfileCapture,
) -> @spec.ModelIdentity {
  self.identity
}

///|
pub fn LunaKernelProfileCapture::target(
  self : LunaKernelProfileCapture,
) -> @catalog.DeviceTarget {
  self.target
}

///|
pub fn LunaKernelProfileCapture::workload_digest(
  self : LunaKernelProfileCapture,
) -> LunaKernelProfileDigest {
  self.workload
}

///|
pub fn LunaKernelProfileCapture::profiler_digest(
  self : LunaKernelProfileCapture,
) -> LunaKernelProfileDigest {
  self.profiler
}

///|
pub fn LunaKernelProfileCapture::sampled_wall_time_ns(
  self : LunaKernelProfileCapture,
) -> Int64 {
  self.sampled_wall_time_ns
}

///|
pub fn LunaKernelProfileCapture::attributed_self_time_ns(
  self : LunaKernelProfileCapture,
) -> Int64 {
  self.attributed_self_time_ns
}

///|
pub fn LunaKernelProfileCapture::observation_count(
  self : LunaKernelProfileCapture,
) -> Int {
  self.observations.length()
}

///|
pub fn LunaKernelProfileCapture::priority(
  self : LunaKernelProfileCapture,
) -> LunaProfiledOperationPriority {
  self.priority
}

///|
pub fn LunaKernelProfileCapture::canonical_bytes(
  self : LunaKernelProfileCapture,
) -> Bytes {
  self.canonical
}

///|
pub fn LunaKernelProfileCapture::digest(
  self : LunaKernelProfileCapture,
) -> LunaKernelProfileDigest {
  self.digest
}
