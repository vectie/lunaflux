///|
/// Startup-owned reusable storage for one canonical submitted-plan frame.
pub struct PlanFrameBuffer {
  priv worker_limits : @worker_protocol.WorkerProtocolLimits
  priv inference_limits : @inference.InferenceLimits
  priv storage : FixedArray[Byte]
  priv mut epoch : UInt64
  priv mut length : Int
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
    worker_limits,
    inference_limits,
    storage: FixedArray::make(capacity, b'\x00'),
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
