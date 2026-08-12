///|
/// Startup choice represented by one explanation entry.
pub(all) enum DecisionSubject {
  DiagnosticMode
  ModelArchitecture
  PrimaryDevice
  MemoryCeiling
} derive(Debug, Eq)

///|
/// Stable reason vocabulary for an effective startup choice.
pub(all) enum DecisionReason {
  ExplicitConfiguration
  FirstAllowedArchitecture
  FirstVisibleDevice
} derive(Debug, Eq)

///|
/// One immutable, bounded startup explanation.
pub struct Decision {
  subject : DecisionSubject
  reason : DecisionReason
} derive(Debug, Eq)

///|
/// Immutable startup decisions derived from already validated focused records.
pub struct ResolvedPlan {
  diagnostic_mode : @service.DiagnosticMode
  selected_architecture : String
  primary_device : Int
  memory_ceiling_bytes : Int64
  decisions : ReadOnlyArray[Decision]
} derive(Debug, Eq)

///|
/// Resolves deterministic startup choices without probing or materializing a
/// model. Validation remains owned by the focused input packages.
pub fn resolve(
  service : @service.ServiceConfig,
  model : @model.ModelConfig,
  device : @device.DeviceConfig,
) -> ResolvedPlan {
  let diagnostic_mode = service.diagnostic_mode()
  let selected_architecture = model.architecture_allowlist()[0]
  let primary_device = device.visible_devices()[0]
  let decisions : Array[Decision] = [
    { subject: DiagnosticMode, reason: ExplicitConfiguration },
    { subject: ModelArchitecture, reason: FirstAllowedArchitecture },
    { subject: PrimaryDevice, reason: FirstVisibleDevice },
    { subject: MemoryCeiling, reason: ExplicitConfiguration },
  ]
  {
    diagnostic_mode,
    selected_architecture,
    primary_device,
    memory_ceiling_bytes: device.memory_ceiling_bytes(),
    decisions: ReadOnlyArray::from_array(decisions),
  }
}

///|
pub fn ResolvedPlan::diagnostic_mode(
  self : ResolvedPlan,
) -> @service.DiagnosticMode {
  self.diagnostic_mode
}

///|
pub fn Decision::subject(self : Decision) -> DecisionSubject {
  self.subject
}

///|
pub fn Decision::reason(self : Decision) -> DecisionReason {
  self.reason
}

///|
pub fn ResolvedPlan::selected_architecture(self : ResolvedPlan) -> String {
  self.selected_architecture
}

///|
pub fn ResolvedPlan::primary_device(self : ResolvedPlan) -> Int {
  self.primary_device
}

///|
pub fn ResolvedPlan::memory_ceiling_bytes(self : ResolvedPlan) -> Int64 {
  self.memory_ceiling_bytes
}

///|
pub fn ResolvedPlan::decisions(self : ResolvedPlan) -> ReadOnlyArray[Decision] {
  self.decisions
}
