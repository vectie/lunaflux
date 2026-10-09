///|
/// Startup analysis only. Stream FIFO may impose additional ordering; this
/// query answers only dependencies guaranteed by the effect graph itself.
pub fn Program::happens_before(
  self : Program,
  earlier : Int,
  later : Int,
) -> Bool {
  if earlier < 0 || later >= self.steps.length() || earlier >= later {
    return false
  }
  let reachable = FixedArray::make(later + 1, false)
  reachable[later] = true
  for index = later; index > earlier; index = index - 1 {
    if reachable[index] {
      for dependency in self.steps[index].dependencies {
        reachable[dependency] = true
      }
    }
  }
  reachable[earlier]
}

///|
/// Whether every access through this view or any physical alias completes
/// before a candidate step. A source-order last-use index alone is insufficient.
pub fn Program::can_reuse_before(
  self : Program,
  buffer : Int,
  step : Int,
) -> Bool {
  guard self.buffers.get(buffer) is Some(target) else { return false }
  if step < 0 || step >= self.steps.length() {
    return false
  }
  for user in self.lifetimes[buffer].users {
    if user >= step {
      return false
    }
  }
  for index, operation in self.steps {
    // Uses at/after the candidate belong to the next contents of the region.
    if index >= step {
      break
    }
    for access in operation.accesses {
      let other = self.buffers[access.buffer]
      if target.arena == other.arena &&
        target.offset < other.offset + other.bytes &&
        other.offset < target.offset + target.bytes &&
        !self.happens_before(index, step) {
        return false
      }
    }
  }
  true
}
