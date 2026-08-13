///|
/// Direction of one startup handshake frame.
pub(all) enum StartupFrameKind {
  Configure
  Ready
} derive(Debug, Eq)

///|
/// Content identity of the complete admitted worker bootstrap evidence.
pub struct WorkerBootstrapDigest(String) derive(Debug, Eq)

///|
/// Exact model, generation, sequence domain, and bounded runtime envelope.
pub struct WorkerStartupContract {
  priv identity : @spec.ModelIdentity
  priv bootstrap_digest : WorkerBootstrapDigest
  priv model_generation_value : UInt64
  priv worker_limits : @worker_protocol.WorkerProtocolLimits
  priv inference_limits : @inference.InferenceLimits
  priv has_predecessor : Bool
  priv predecessor_value : UInt64
} derive(Debug, Eq)

///|
/// Startup-owned fixed storage for one startup handshake frame.
pub struct StartupFrameBuffer {
  priv storage : FixedArray[Byte]
  priv mut epoch : UInt64
  priv mut length : Int
}

///|
/// Authenticated immutable view of one startup-frame epoch.
#valtype
pub struct ValidatedStartupFrame(StartupFrameBuffer, UInt64)

///|
pub fn WorkerStartupContract::new(
  identity~ : @spec.ModelIdentity,
  bootstrap_digest~ : WorkerBootstrapDigest,
  model_generation_value~ : UInt64,
  worker_limits~ : @worker_protocol.WorkerProtocolLimits,
  inference_limits~ : @inference.InferenceLimits,
  has_predecessor~ : Bool,
  predecessor_value~ : UInt64,
) -> WorkerStartupContract raise WorkerWireError {
  if model_generation_value == 0UL ||
    (
      has_predecessor &&
      (predecessor_value == 0UL || predecessor_value == 0xffffffffffffffffUL)
    ) ||
    (!has_predecessor && predecessor_value != 0UL) {
    raise InvalidFrame(StartupIdentity)
  }
  {
    identity,
    bootstrap_digest,
    model_generation_value,
    worker_limits,
    inference_limits,
    has_predecessor,
    predecessor_value,
  }
}

///|
pub fn WorkerBootstrapDigest::from_sha256(
  value : String,
) -> WorkerBootstrapDigest raise WorkerWireError {
  if value.length() != 64 {
    raise InvalidFrame(StartupIdentity)
  }
  for character in value {
    if !(character is ('0'..='9' | 'a'..='f')) {
      raise InvalidFrame(StartupIdentity)
    }
  }
  WorkerBootstrapDigest(value)
}

///|
pub fn WorkerBootstrapDigest::as_hex(self : WorkerBootstrapDigest) -> String {
  self.0
}

///|
pub fn WorkerStartupContract::identity(
  self : WorkerStartupContract,
) -> @spec.ModelIdentity {
  self.identity
}

///|
pub fn WorkerStartupContract::bootstrap_digest(
  self : WorkerStartupContract,
) -> WorkerBootstrapDigest {
  self.bootstrap_digest
}

///|
pub fn WorkerStartupContract::model_generation_value(
  self : WorkerStartupContract,
) -> UInt64 {
  self.model_generation_value
}

///|
pub fn WorkerStartupContract::worker_limits(
  self : WorkerStartupContract,
) -> @worker_protocol.WorkerProtocolLimits {
  self.worker_limits
}

///|
pub fn WorkerStartupContract::inference_limits(
  self : WorkerStartupContract,
) -> @inference.InferenceLimits {
  self.inference_limits
}

///|
pub fn WorkerStartupContract::has_predecessor(
  self : WorkerStartupContract,
) -> Bool {
  self.has_predecessor
}

///|
pub fn WorkerStartupContract::predecessor_value(
  self : WorkerStartupContract,
) -> UInt64 {
  self.predecessor_value
}

///|
pub fn StartupFrameBuffer::new() -> StartupFrameBuffer {
  { storage: FixedArray::make(344, b'\x00'), epoch: 0UL, length: 0 }
}

///|
pub fn StartupFrameBuffer::capacity(self : StartupFrameBuffer) -> Int {
  self.storage.length()
}
