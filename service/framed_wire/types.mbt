///|
const ABSOLUTE_MAX_FRAME_BYTES : Int = 16777216

///|
const REQUEST_HEADER_BYTES : Int = 256

///|
const EVENT_HEADER_BYTES : Int = 208

///|
pub(all) enum FramedWireFrameKind {
  RequestFrame
  EventFrame
} derive(Debug, Eq)

///|
pub(all) enum FramedWireRule {
  Magic
  Version
  Kind
  Length
  Checksum
  Flags
  Reserved
  Identity
  Scalar
  Input
  StopTable
  Payload
  CanonicalEncoding
} derive(Debug, Eq)

///|
pub(all) enum FramedWireLimitRule {
  FrameBytes
} derive(Debug, Eq)

///|
pub(all) suberror FramedWireError {
  InvalidLimit(FramedWireLimitRule)
  InvalidFrame(frame~ : FramedWireFrameKind, rule~ : FramedWireRule)
  FrameTooLarge(frame~ : FramedWireFrameKind)
  DestinationTooSmall
  EventCreditUnavailable
  RequestReaderIncomplete
  RequestReaderPoisoned
  StaleFrame
} derive(Debug, Eq)

///|
pub struct FramedWireLimits {
  priv inference : @inference.InferenceLimits
  priv max_frame_bytes : Int
} derive(Debug, Eq)

///|
pub fn FramedWireLimits::new(
  inference : @inference.InferenceLimits,
  max_frame_bytes~ : Int,
) -> FramedWireLimits raise FramedWireError {
  if max_frame_bytes < REQUEST_HEADER_BYTES ||
    max_frame_bytes > ABSOLUTE_MAX_FRAME_BYTES {
    raise InvalidLimit(FrameBytes)
  }
  { inference, max_frame_bytes }
}

///|
pub fn FramedWireLimits::inference_limits(
  self : FramedWireLimits,
) -> @inference.InferenceLimits {
  self.inference
}

///|
pub fn FramedWireLimits::max_frame_bytes(self : FramedWireLimits) -> Int {
  self.max_frame_bytes
}

///|
pub struct RequestFrameBuffer {
  priv limits : FramedWireLimits
  priv storage : FixedArray[Byte]
  priv scratch : FixedArray[Byte]
  priv mut request : @inference.GenerateRequest?
  priv mut epoch : UInt64
  priv mut length : Int
}

///|
#valtype
pub struct ValidatedRequestFrame {
  priv owner_ref : RequestFrameBuffer
  priv epoch : UInt64
}

///|
/// One fixed-capacity incremental owner for exactly one canonical request
/// frame. Prefix or frame rejection permanently poisons the owner.
pub struct IncrementalRequestReader {
  priv frame_buffer : RequestFrameBuffer
  priv mut frame : ValidatedRequestFrame?
  priv mut length : Int
  priv mut declared_length : Int
  priv mut state : Int
}

///|
pub struct EventFrameBuffer {
  priv limits : FramedWireLimits
  priv storage : FixedArray[Byte]
  priv scratch : FixedArray[Byte]
  priv mut event : @inference.StreamEvent?
  priv mut epoch : UInt64
  priv mut length : Int
}

///|
#valtype
pub struct ValidatedEventFrame {
  priv owner_ref : EventFrameBuffer
  priv epoch : UInt64
}
