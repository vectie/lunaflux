///|
/// One bounded profiler counter. Values are observations only and grant no
/// device, launch, scheduler, or runtime authority.
pub(all) enum LunaPagedProfileCounterKind {
  KernelLaunchCount
  DramReadBytes
  DramWriteBytes
  L2ReadBytes
  L2WriteBytes
} derive(Debug, Eq)

///|
fn LunaPagedProfileCounterKind::stable_int(
  self : LunaPagedProfileCounterKind,
) -> Int {
  match self {
    KernelLaunchCount => 1
    DramReadBytes => 2
    DramWriteBytes => 3
    L2ReadBytes => 4
    L2WriteBytes => 5
  }
}

///|
pub struct LunaPagedProfileCounter {
  priv kind : LunaPagedProfileCounterKind
  priv value : Int64
} derive(Debug, Eq)

///|
pub fn LunaPagedProfileCounter::new(
  kind~ : LunaPagedProfileCounterKind,
  value~ : Int64,
) -> LunaPagedProfileCounter raise LunaKernelProfileError {
  if value < 0L || value > 9_000_000_000_000_000L {
    raise LunaKernelProfileInvalid(Counter)
  }
  { kind, value }
}

///|
pub fn LunaPagedProfileCounter::kind(
  self : LunaPagedProfileCounter,
) -> LunaPagedProfileCounterKind {
  self.kind
}

///|
pub fn LunaPagedProfileCounter::value(self : LunaPagedProfileCounter) -> Int64 {
  self.value
}

///|
/// One operation's measured self/inclusive time for one exact paged shape.
pub struct LunaPagedKernelProfileObservation {
  priv operation_id : @plan.OperationId
  priv shape : LunaPagedProfileShape
  priv call_count : Int64
  priv self_time_ns : Int64
  priv inclusive_time_ns : Int64
  priv counters : ReadOnlyArray[LunaPagedProfileCounter]
} derive(Debug, Eq)

///|
pub fn LunaPagedKernelProfileObservation::new(
  operation_id~ : @plan.OperationId,
  shape~ : LunaPagedProfileShape,
  call_count~ : Int64,
  self_time_ns~ : Int64,
  inclusive_time_ns~ : Int64,
  counters? : ArrayView[LunaPagedProfileCounter] = [],
) -> LunaPagedKernelProfileObservation raise LunaKernelProfileError {
  if call_count <= 0L ||
    call_count > 1_000_000_000L ||
    self_time_ns <= 0L ||
    self_time_ns > LUNA_KERNEL_PROFILE_MAX_NANOS ||
    inclusive_time_ns < self_time_ns ||
    inclusive_time_ns > LUNA_KERNEL_PROFILE_MAX_NANOS {
    raise LunaKernelProfileInvalid(Timing)
  }
  if counters.length() > 32 {
    raise LunaKernelProfileInvalid(Counter)
  }
  for index, counter in counters {
    if index > 0 &&
      counter.kind().stable_int() <= counters[index - 1].kind().stable_int() {
      raise LunaKernelProfileInvalid(CounterOrder)
    }
  }
  {
    operation_id,
    shape,
    call_count,
    self_time_ns,
    inclusive_time_ns,
    counters: ReadOnlyArray::from_array(counters),
  }
}

///|
pub fn LunaPagedKernelProfileObservation::operation_id(
  self : LunaPagedKernelProfileObservation,
) -> @plan.OperationId {
  self.operation_id
}

///|
pub fn LunaPagedKernelProfileObservation::shape(
  self : LunaPagedKernelProfileObservation,
) -> LunaPagedProfileShape {
  self.shape
}

///|
pub fn LunaPagedKernelProfileObservation::call_count(
  self : LunaPagedKernelProfileObservation,
) -> Int64 {
  self.call_count
}

///|
pub fn LunaPagedKernelProfileObservation::self_time_ns(
  self : LunaPagedKernelProfileObservation,
) -> Int64 {
  self.self_time_ns
}

///|
pub fn LunaPagedKernelProfileObservation::inclusive_time_ns(
  self : LunaPagedKernelProfileObservation,
) -> Int64 {
  self.inclusive_time_ns
}

///|
pub fn LunaPagedKernelProfileObservation::counters(
  self : LunaPagedKernelProfileObservation,
) -> ReadOnlyArray[LunaPagedProfileCounter] {
  self.counters
}

///|
/// Immutable paged priority evidence. It grants no profiler, compiler, kernel
/// promotion, device, or serving authority.
pub struct LunaPagedKernelProfileCapture {
  priv identity : @spec.ModelIdentity
  priv target : @catalog.DeviceTarget
  priv workload : LunaKernelProfileDigest
  priv profiler : LunaKernelProfileDigest
  priv trace : LunaKernelProfileDigest
  priv contract : LunaPagedProfileContract
  priv sampled_wall_time_ns : Int64
  priv attributed_self_time_ns : Int64
  priv observations : ReadOnlyArray[LunaPagedKernelProfileObservation]
  priv priority : LunaProfiledOperationPriority
  priv canonical : Bytes
  priv digest : LunaKernelProfileDigest
} derive(Debug, Eq)

///|
pub fn LunaPagedKernelProfileCapture::identity(
  self : LunaPagedKernelProfileCapture,
) -> @spec.ModelIdentity {
  self.identity
}

///|
pub fn LunaPagedKernelProfileCapture::target(
  self : LunaPagedKernelProfileCapture,
) -> @catalog.DeviceTarget {
  self.target
}

///|
pub fn LunaPagedKernelProfileCapture::workload_digest(
  self : LunaPagedKernelProfileCapture,
) -> LunaKernelProfileDigest {
  self.workload
}

///|
pub fn LunaPagedKernelProfileCapture::profiler_digest(
  self : LunaPagedKernelProfileCapture,
) -> LunaKernelProfileDigest {
  self.profiler
}

///|
pub fn LunaPagedKernelProfileCapture::trace_digest(
  self : LunaPagedKernelProfileCapture,
) -> LunaKernelProfileDigest {
  self.trace
}

///|
pub fn LunaPagedKernelProfileCapture::contract(
  self : LunaPagedKernelProfileCapture,
) -> LunaPagedProfileContract {
  self.contract
}

///|
pub fn LunaPagedKernelProfileCapture::sampled_wall_time_ns(
  self : LunaPagedKernelProfileCapture,
) -> Int64 {
  self.sampled_wall_time_ns
}

///|
pub fn LunaPagedKernelProfileCapture::attributed_self_time_ns(
  self : LunaPagedKernelProfileCapture,
) -> Int64 {
  self.attributed_self_time_ns
}

///|
pub fn LunaPagedKernelProfileCapture::observation_count(
  self : LunaPagedKernelProfileCapture,
) -> Int {
  self.observations.length()
}

///|
pub fn LunaPagedKernelProfileCapture::priority(
  self : LunaPagedKernelProfileCapture,
) -> LunaProfiledOperationPriority {
  self.priority
}

///|
pub fn LunaPagedKernelProfileCapture::canonical_bytes(
  self : LunaPagedKernelProfileCapture,
) -> Bytes {
  self.canonical
}

///|
pub fn LunaPagedKernelProfileCapture::digest(
  self : LunaPagedKernelProfileCapture,
) -> LunaKernelProfileDigest {
  self.digest
}
