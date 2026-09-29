///|
/// Target-independent execution choices. Geometry is supplied by a backend's
/// supported candidate set, not inferred from a model name or device string.
pub struct Policy {
  priv id : String
  priv consumers : Int
  priv accumulators : Int
  priv stages : Int
  priv tiles : Int
  priv fragments : Int
  priv row_tiles : Int
  priv head_tiles : Int
} derive(Eq, Debug)

///|
pub extend Policy with Eq::{equal, not_equal}

///|
pub extend Policy with @debug.Debug::{to_repr}

///|
pub fn Policy::new(
  id~ : String,
  consumer_groups~ : Int,
  live_accumulators~ : Int,
  operand_stages~ : Int,
  transfer_tiles~ : Int,
  fragment_stages~ : Int,
  row_tiles? : Int = 1,
  head_tiles? : Int = 1,
) -> Policy? {
  guard valid_label(id) &&
    consumer_groups > 0 &&
    live_accumulators > 0 &&
    operand_stages > 0 &&
    transfer_tiles > 0 &&
    fragment_stages > 0 &&
    fragment_stages <= transfer_tiles &&
    row_tiles > 0 &&
    head_tiles > 0 else {
    return None
  }
  Some({
    id,
    consumers: consumer_groups,
    accumulators: live_accumulators,
    stages: operand_stages,
    tiles: transfer_tiles,
    fragments: fragment_stages,
    row_tiles,
    head_tiles,
  })
}

///|
pub fn Policy::id(self : Policy) -> String {
  self.id
}

///|
pub fn Policy::consumer_groups(self : Policy) -> Int {
  self.consumers
}

///|
pub fn Policy::live_accumulators(self : Policy) -> Int {
  self.accumulators
}

///|
pub fn Policy::operand_stages(self : Policy) -> Int {
  self.stages
}

///|
pub fn Policy::transfer_tiles(self : Policy) -> Int {
  self.tiles
}

///|
pub fn Policy::fragment_stages(self : Policy) -> Int {
  self.fragments
}

///|
pub fn Policy::row_tiles(self : Policy) -> Int {
  self.row_tiles
}

///|
pub fn Policy::head_tiles(self : Policy) -> Int {
  self.head_tiles
}

///|
pub struct Scope {
  priv device : String
  priv toolchain : String
  priv shape : String
  priv program : String
} derive(Eq, Debug)

///|
pub extend Scope with Eq::{equal, not_equal}

///|
pub extend Scope with @debug.Debug::{to_repr}

///|
/// Exact offline identity: the caller includes dtype/layout/shape and the
/// compiler/source revision in shape/program. No parsing of hardware identity.
pub fn Scope::new(
  device~ : String,
  toolchain~ : String,
  shape~ : String,
  program~ : String,
) -> Scope? {
  guard valid_label(device) &&
    valid_label(toolchain) &&
    valid_label(shape) &&
    valid_label(program) else {
    return None
  }
  Some({ device, toolchain, shape, program, })
}

///|
pub fn Scope::identity(self : Scope) -> (String, String, String, String) {
  (self.device, self.toolchain, self.shape, self.program)
}

///|
fn valid_label(value : String) -> Bool {
  value.length() > 0 &&
  value.length() <= 1024 &&
  value.iter().all(c => c != '\n' && c != '\r' && c != '\t')
}

///|
/// The backend owns hardware/launch limits. None means the caller deliberately
/// does not impose that resource constraint; it is not an invented measurement.
pub struct ResourceBudget {
  priv threads : Int
  priv shared : Int64
  priv registers : Int?
  priv residency : Int?
} derive(Eq, Debug)

///|
pub extend ResourceBudget with Eq::{equal, not_equal}

///|
pub extend ResourceBudget with @debug.Debug::{to_repr}

///|
pub fn ResourceBudget::new(
  max_threads~ : Int,
  max_shared_bytes~ : Int64,
  max_registers_per_thread~ : Int?,
  min_resident_groups~ : Int?,
) -> ResourceBudget? {
  guard max_threads > 0 && max_shared_bytes >= 0L else { return None }
  if max_registers_per_thread is Some(n) && n <= 0 {
    return None
  }
  if min_resident_groups is Some(n) && n <= 0 {
    return None
  }
  Some({
    threads: max_threads,
    shared: max_shared_bytes,
    registers: max_registers_per_thread,
    residency: min_resident_groups,
  })
}

///|
/// Offline target limits may narrow, never relax, backend launch constraints.
pub fn ResourceBudget::intersect(
  self : ResourceBudget,
  other : ResourceBudget,
) -> ResourceBudget {
  {
    threads: Int::min(self.threads, other.threads),
    shared: Int64::min(self.shared, other.shared),
    registers: match (self.registers, other.registers) {
      (Some(a), Some(b)) => Some(Int::min(a, b))
      (Some(a), None) | (None, Some(a)) => Some(a)
      _ => None
    },
    residency: match (self.residency, other.residency) {
      (Some(a), Some(b)) => Some(Int::max(a, b))
      (Some(a), None) | (None, Some(a)) => Some(a)
      _ => None
    },
  }
}

///|
pub struct Candidate {
  priv policy : Policy
  priv artifact : String
  priv threads : Int
  priv shared : Int64
  priv registers : Int?
  priv residency : Int?
} derive(Eq, Debug)

///|
pub extend Candidate with Eq::{equal, not_equal}

///|
pub extend Candidate with @debug.Debug::{to_repr}

///|
/// Resource values here are conservative compilation bounds, not runtime
/// measurements. Unknown register/residency bounds cannot satisfy a constraint.
pub fn Candidate::new(
  policy : Policy,
  artifact~ : String,
  threads~ : Int,
  shared_bytes~ : Int64,
  registers_per_thread~ : Int?,
  resident_groups~ : Int?,
) -> Candidate? {
  guard valid_label(artifact) && threads > 0 && shared_bytes >= 0L else {
    return None
  }
  if registers_per_thread is Some(n) && n <= 0 {
    return None
  }
  if resident_groups is Some(n) && n <= 0 {
    return None
  }
  Some({
    policy,
    artifact,
    threads,
    shared: shared_bytes,
    registers: registers_per_thread,
    residency: resident_groups,
  })
}

///|
pub struct Observation {
  priv scope : Scope
  priv policy_id : String
  priv artifact : String
  priv latency : Int64
  priv samples : Int
  priv registers : Int
  priv residency : Int
} derive(Eq, Debug)

///|
pub extend Observation with Eq::{equal, not_equal}

///|
pub extend Observation with @debug.Debug::{to_repr}

///|
pub fn Observation::new(
  scope : Scope,
  policy_id~ : String,
  artifact~ : String,
  latency_ns~ : Int64,
  samples~ : Int,
  registers_per_thread~ : Int,
  resident_groups~ : Int,
) -> Observation? {
  guard valid_label(policy_id) &&
    valid_label(artifact) &&
    latency_ns > 0L &&
    samples >= 3 &&
    registers_per_thread > 0 &&
    resident_groups > 0 else {
    return None
  }
  Some({
    scope,
    policy_id,
    artifact,
    latency: latency_ns,
    samples,
    registers: registers_per_thread,
    residency: resident_groups,
  })
}

///|
pub(all) enum SelectionOrigin {
  Measured(latency_ns~ : Int64, samples~ : Int)
  UnmeasuredPreferred
  UnmeasuredEligible
} derive(Eq, Debug)

///|
pub extend SelectionOrigin with Eq::{equal, not_equal}

///|
pub extend SelectionOrigin with @debug.Debug::{to_repr}

///|
pub struct Selection {
  priv candidate : Candidate
  priv origin : SelectionOrigin
} derive(Eq, Debug)

///|
pub extend Selection with Eq::{equal, not_equal}

///|
pub extend Selection with @debug.Debug::{to_repr}

///|
pub fn Selection::policy(self : Selection) -> Policy {
  self.candidate.policy
}

///|
pub fn Selection::origin(self : Selection) -> SelectionOrigin {
  self.origin
}

///|
pub fn Selection::artifact(self : Selection) -> String {
  self.candidate.artifact
}
