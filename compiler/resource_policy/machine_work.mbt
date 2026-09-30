///|
/// Scope/artifact-bound machine feedback, collected offline from the selected
/// executable. These are counters, not estimated latency. A caller may impose
/// a search envelope, but the winner still minimizes measured completion time.
pub struct MachineWork {
  priv instructions : Int64
  priv local_bytes : Int64
  priv barriers : Int64?
} derive(Eq, Debug)

///|
pub fn MachineWork::new(
  instructions~ : Int64,
  local_bytes~ : Int64,
  barriers? : Int64? = None,
) -> MachineWork? {
  if instructions < 0L ||
    local_bytes < 0L ||
    (barriers is Some(value) && value < 0L) {
    None
  } else {
    Some({ instructions, local_bytes, barriers, })
  }
}

///|
pub fn MachineWork::counts(self : MachineWork) -> (Int64, Int64, Int64?) {
  (self.instructions, self.local_bytes, self.barriers)
}

///|
fn MachineWork::fits(self : MachineWork, limits : MachineWork) -> Bool {
  self.instructions <= limits.instructions &&
  self.local_bytes <= limits.local_bytes &&
  (match (self.barriers, limits.barriers) {
    (_, None) => true
    (Some(actual), Some(maximum)) => actual <= maximum
    (None, Some(_)) => false
  })
}

///|
fn MachineWork::intersect(
  self : MachineWork,
  other : MachineWork,
) -> MachineWork {
  {
    instructions: Int64::min(self.instructions, other.instructions),
    local_bytes: Int64::min(self.local_bytes, other.local_bytes),
    barriers: match (self.barriers, other.barriers) {
      (Some(left), Some(right)) => Some(Int64::min(left, right))
      (Some(value), None) | (None, Some(value)) => Some(value)
      (None, None) => None
    },
  }
}
