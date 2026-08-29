///|
/// Startup-owned reusable storage for one canonical submitted-plan frame.
pub struct PlanFrameBuffer {
  priv identity : PlanFrameOwnerIdentity
  priv worker_limits : @worker_protocol.WorkerProtocolLimits
  priv inference_limits : @inference.InferenceLimits
  priv storage : FixedArray[Byte]
  priv frame_offset : Int
  priv validation_requests : FixedArray[UInt64]
  priv validation_slots : FixedArray[Int]
  priv mut epoch : UInt64
  priv mut length : Int
}

///|
/// Opaque unforgeable identity of one startup-owned plan-frame buffer. The
/// value exposes no storage or mutation authority.
pub struct PlanFrameOwnerIdentity {
  priv marker : Ref[Int]
}

///|
/// Authenticated immutable view of one exact encoded or received frame epoch.
#valtype
pub struct ValidatedPlanFrame(PlanFrameBuffer, UInt64)

///|
/// Opaque prefill-row view tied to one exact frame epoch.
#valtype
pub struct PlanFramePrefillRow(PlanFrameBuffer, UInt64, Int)

///|
/// Opaque decode-row view tied to one exact frame epoch.
#valtype
pub struct PlanFrameDecodeRow(PlanFrameBuffer, UInt64, Int)

///|
pub fn PlanFrameBuffer::new(
  worker_limits : @worker_protocol.WorkerProtocolLimits,
  inference_limits : @inference.InferenceLimits,
) -> PlanFrameBuffer raise WorkerWireError {
  let capacity = plan_frame_byte_capacity(worker_limits)
  {
    identity: { marker: Ref(0) },
    worker_limits,
    inference_limits,
    storage: FixedArray::make(capacity, b'\x00'),
    frame_offset: 0,
    validation_requests: FixedArray::make(worker_limits.max_plan_rows(), 0UL),
    validation_slots: FixedArray::make(worker_limits.max_plan_rows(), -1),
    epoch: 0UL,
    length: 0,
  }
}

///|
/// Transfers a caller-owned transport buffer into a plan-frame owner. The
/// caller must retain no independently writable alias; production owners keep
/// the only alias private and use it solely for an immutable child write while
/// the corresponding frame epoch is live.
pub fn PlanFrameBuffer::new_transport_owned(
  worker_limits : @worker_protocol.WorkerProtocolLimits,
  inference_limits : @inference.InferenceLimits,
  transport : FixedArray[Byte],
  frame_offset~ : Int,
) -> PlanFrameBuffer raise WorkerWireError {
  let capacity = plan_frame_byte_capacity(worker_limits)
  if frame_offset < 0 ||
    frame_offset > transport.length() ||
    capacity > transport.length() - frame_offset {
    raise InvalidLimit(PlanFrameBytes)
  }
  {
    identity: { marker: Ref(0) },
    worker_limits,
    inference_limits,
    storage: transport,
    frame_offset,
    validation_requests: FixedArray::make(worker_limits.max_plan_rows(), 0UL),
    validation_slots: FixedArray::make(worker_limits.max_plan_rows(), -1),
    epoch: 0UL,
    length: 0,
  }
}

///|
pub fn PlanFrameBuffer::capacity(self : PlanFrameBuffer) -> Int {
  self.storage.length() - self.frame_offset
}

///|
/// Byte offset of this immutable frame within its privately owned transport.
pub fn ValidatedPlanFrame::transport_offset(
  self : ValidatedPlanFrame,
) -> Int raise WorkerWireError {
  self.owner().frame_offset
}

///|
/// Rederives the current immutable frame view without re-encoding it.
pub fn PlanFrameBuffer::current(
  self : PlanFrameBuffer,
) -> ValidatedPlanFrame raise WorkerWireError {
  if self.epoch == 0UL || self.length == 0 {
    raise StaleFrame
  }
  ValidatedPlanFrame(self, self.epoch)
}

///|
fn ValidatedPlanFrame::owner(
  self : ValidatedPlanFrame,
) -> PlanFrameBuffer raise WorkerWireError {
  if self.1 == 0UL || self.0.epoch != self.1 || self.0.length == 0 {
    raise StaleFrame
  }
  self.0
}

///|
pub fn ValidatedPlanFrame::length(
  self : ValidatedPlanFrame,
) -> Int raise WorkerWireError {
  self.owner().length
}

///|
pub fn ValidatedPlanFrame::owner_identity(
  self : ValidatedPlanFrame,
) -> PlanFrameOwnerIdentity raise WorkerWireError {
  self.owner().identity
}

///|
pub fn ValidatedPlanFrame::epoch_value(
  self : ValidatedPlanFrame,
) -> UInt64 raise WorkerWireError {
  ignore(self.owner())
  self.1
}

///|
pub fn ValidatedPlanFrame::matches_owner_epoch(
  self : ValidatedPlanFrame,
  owner : PlanFrameOwnerIdentity,
  epoch : UInt64,
) -> Bool raise WorkerWireError {
  let actual = self.owner()
  ignore(actual.identity.marker.val)
  physical_equal(actual.identity.marker, owner.marker) && self.1 == epoch
}
