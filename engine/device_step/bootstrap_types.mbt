///|
pub(all) enum DeviceBootstrapRule {
  Capacity
  Identity
  DeviceOrdinal
  Target
  CatalogVersion
  Scope
  Artifact
  Blueprint
  RuntimeEnvelope
  CanonicalBytes
} derive(Debug, Eq)

///|
pub(all) suberror DeviceBootstrapError {
  InvalidBootstrap(DeviceBootstrapRule)
  BootstrapWire(@worker_wire.WorkerWireError)
} derive(Debug, Eq)

///|
/// Bounded startup limits for canonical worker-bootstrap evidence.
pub struct DeviceBootstrapLimits {
  priv max_modules : Int
  priv max_entry_points : Int
  priv max_steps : Int
  priv max_operands_per_step : Int
  priv max_symbol_bytes : Int
  priv max_canonical_bytes : Int
}

///|
/// Immutable identity and bounded summary of one admitted device bootstrap.
pub struct DeviceWorkerBootstrapManifest {
  priv digest : @worker_wire.WorkerBootstrapDigest
  priv identity : @spec.ModelIdentity
  priv device_ordinal : Int
  priv target : @catalog.DeviceTarget
  priv catalog_version : @catalog.CatalogVersion
  priv limits : DeviceStepLimits
  priv module_count : Int
  priv entry_point_count : Int
  priv step_count : Int
}

///|
pub fn DeviceBootstrapLimits::new(
  max_modules~ : Int,
  max_entry_points~ : Int,
  max_steps~ : Int,
  max_operands_per_step~ : Int,
  max_symbol_bytes~ : Int,
  max_canonical_bytes~ : Int,
) -> DeviceBootstrapLimits raise DeviceBootstrapError {
  if max_modules <= 0 ||
    max_entry_points <= 0 ||
    max_steps <= 0 ||
    max_operands_per_step <= 0 ||
    max_symbol_bytes <= 0 ||
    max_canonical_bytes < 256 ||
    max_modules > 65536 ||
    max_entry_points > 65536 ||
    max_steps > 65536 ||
    max_operands_per_step > 256 ||
    max_symbol_bytes > 4096 ||
    max_canonical_bytes > 16777216 {
    raise InvalidBootstrap(Capacity)
  }
  {
    max_modules,
    max_entry_points,
    max_steps,
    max_operands_per_step,
    max_symbol_bytes,
    max_canonical_bytes,
  }
}

///|
pub fn DeviceWorkerBootstrapManifest::digest(
  self : DeviceWorkerBootstrapManifest,
) -> @worker_wire.WorkerBootstrapDigest {
  self.digest
}

///|
pub fn DeviceWorkerBootstrapManifest::identity(
  self : DeviceWorkerBootstrapManifest,
) -> @spec.ModelIdentity {
  self.identity
}

///|
/// Returns the exact ordinal within the device set visible to the worker
/// process. This value is not a persistent hardware UUID.
pub fn DeviceWorkerBootstrapManifest::device_ordinal(
  self : DeviceWorkerBootstrapManifest,
) -> Int {
  self.device_ordinal
}

///|
pub fn DeviceWorkerBootstrapManifest::target(
  self : DeviceWorkerBootstrapManifest,
) -> @catalog.DeviceTarget {
  self.target
}

///|
pub fn DeviceWorkerBootstrapManifest::catalog_version(
  self : DeviceWorkerBootstrapManifest,
) -> @catalog.CatalogVersion {
  self.catalog_version
}

///|
/// Returns the exact descriptor and scalar envelope included in the bootstrap
/// digest and required to contain the startup protocol envelopes.
pub fn DeviceWorkerBootstrapManifest::device_step_limits(
  self : DeviceWorkerBootstrapManifest,
) -> DeviceStepLimits {
  self.limits
}

///|
pub fn DeviceWorkerBootstrapManifest::module_count(
  self : DeviceWorkerBootstrapManifest,
) -> Int {
  self.module_count
}

///|
pub fn DeviceWorkerBootstrapManifest::entry_point_count(
  self : DeviceWorkerBootstrapManifest,
) -> Int {
  self.entry_point_count
}

///|
pub fn DeviceWorkerBootstrapManifest::step_count(
  self : DeviceWorkerBootstrapManifest,
) -> Int {
  self.step_count
}

///|
/// Constructs the only process-startup contract for this admitted bootstrap.
pub fn DeviceWorkerBootstrapManifest::worker_startup_contract(
  self : DeviceWorkerBootstrapManifest,
  model_generation : @worker_protocol.ModelPlanGeneration,
  worker_limits : @worker_protocol.WorkerProtocolLimits,
  inference_limits : @inference.InferenceLimits,
  has_predecessor~ : Bool,
  predecessor_value~ : UInt64,
) -> @worker_wire.WorkerStartupContract raise DeviceBootstrapError {
  let device = self.limits
  if worker_limits.max_plan_rows() > device.max_rows() ||
    worker_limits.max_prefill_rows() > device.max_rows() ||
    worker_limits.max_decode_rows() > device.max_rows() ||
    worker_limits.max_plan_tokens() > device.max_tokens() ||
    worker_limits.max_plan_pages() > device.max_pages() ||
    worker_limits.max_sequence_tokens() > device.max_sequence_tokens() ||
    worker_limits.max_token_id() > device.max_token_id() ||
    inference_limits.max_context_tokens() > device.max_sequence_tokens() ||
    inference_limits.max_context_tokens() > worker_limits.max_sequence_tokens() ||
    inference_limits.max_token_id() > device.max_token_id() ||
    inference_limits.max_token_id() > worker_limits.max_token_id() {
    raise InvalidBootstrap(RuntimeEnvelope)
  }
  @worker_wire.WorkerStartupContract::new(
    identity=self.identity,
    bootstrap_digest=self.digest,
    device_ordinal=self.device_ordinal,
    model_generation_value=model_generation.value(),
    worker_limits~,
    inference_limits~,
    has_predecessor~,
    predecessor_value~,
  ) catch {
    error => raise BootstrapWire(error)
  }
}
