///|
pub(all) enum WorkerWireLimitRule {
  PlanFrameBytes
  CompletionFrameBytes
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
  CompletionEntry
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
  CompletionFrameWriterAlreadyOpen
  CompletionFrameWriterNotOpen
  CompletionFrameWriterPlanMismatch
  CompletionFrameWriterOrderMismatch
  Protocol(@worker_protocol.WorkerProtocolError)
} derive(Debug, Eq)
