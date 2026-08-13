///|
/// Startup-owned reusable storage for one canonical submitted-plan frame.
pub struct PlanFrameBuffer {
  priv identity : PlanFrameOwnerIdentity
  priv worker_limits : @worker_protocol.WorkerProtocolLimits
  priv inference_limits : @inference.InferenceLimits
  priv storage : FixedArray[Byte]
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
    validation_requests: FixedArray::make(worker_limits.max_plan_rows(), 0UL),
    validation_slots: FixedArray::make(worker_limits.max_plan_rows(), -1),
    epoch: 0UL,
    length: 0,
  }
}

///|
pub fn PlanFrameBuffer::capacity(self : PlanFrameBuffer) -> Int {
  self.storage.length()
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
