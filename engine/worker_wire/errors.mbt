///|
pub(all) enum WorkerWireLimitRule {
  PlanFrameBytes
  CompletionFrameBytes
  StartupFrameBytes
  BootstrapSourceBytes
} derive(Debug, Eq)

///|
/// Stable, payload-safe reasons why a worker bootstrap source was rejected.
pub(all) enum WorkerBootstrapSourceRule {
  Magic
  Version
  Recipe
  Length
  Reserved
  RootPath
  RelativeLocator
  DuplicateLocator
  Digest
  Target
  KvLayout
  Ceiling
  Utf8
  Integrity
  CanonicalEncoding
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
  StartupIdentity
  Reserved
} derive(Debug, Eq)

///|
/// Payload-safe failures for bounded worker-process frames.
pub(all) suberror WorkerWireError {
  InvalidLimit(WorkerWireLimitRule)
  InvalidFrame(WorkerWireFrameRule)
  InvalidBootstrapSource(WorkerBootstrapSourceRule)
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
