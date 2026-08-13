///|
pub(all) enum WorkerWireLimitRule {
  PlanFrameBytes
} derive(Debug, Eq)

///|
pub(all) enum WorkerWireFrameRule {
  Magic
  Version
  Kind
  Length
  Checksum
  Counts
  TokenTable
  PageTable
  CapabilityTable
  PrefillRow
  DecodeRow
  Sampling
  CompletionSlot
  RequestIdentity
  Reserved
} derive(Debug, Eq)

///|
/// Payload-safe failures for bounded worker-process frames.
pub(all) suberror WorkerWireError {
  InvalidLimit(WorkerWireLimitRule)
  InvalidFrame(WorkerWireFrameRule)
  SourceTooSmall
  DestinationTooSmall
  FrameEpochOverflow
  StaleFrame
  Protocol(@worker_protocol.WorkerProtocolError)
} derive(Debug, Eq)
