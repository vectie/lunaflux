///|
fn release(state : State) -> Transition {
  let next = if state.generation == 2147483647 {
    { ..state, phase: Retired, cancelled: false, }
  } else {
    {
      phase: Free,
      generation: state.generation + 1,
      epoch: 0,
      cancelled: false,
    }
  }
  { state: next, effect: Released, }
}

///|
/// Pure total decision function. Failed transfers retain all reservations until
/// a successful drain. A cancelled spill cannot publish host cache contents.
fn transition(state : State, event : Event) -> Transition {
  match (state.phase, event) {
    (Free, BeginSpill) =>
      {
        state: { ..state, phase: Spilling, epoch: 1, cancelled: false, },
        effect: CopyToHost,
      }
    (Host, BeginRestore) if state.epoch < 2147483647 =>
      {
        state: {
          ..state,
          phase: Restoring,
          epoch: state.epoch + 1,
          cancelled: false,
        },
        effect: CopyToDevice,
      }
    (Spilling | Restoring, Cancel) =>
      { state: { ..state, cancelled: true, }, effect: RetainUntilCompletion, }
    (Spilling, Complete) =>
      if state.cancelled {
        release(state)
      } else {
        { state: { ..state, phase: Host, }, effect: HostReady, }
      }
    (Restoring, Complete) =>
      {
        state: { ..state, phase: Host, cancelled: false, },
        effect: if state.cancelled {
          RestoreCancelled
        } else {
          DeviceReady
        },
      }
    (Spilling | Restoring, Fail) =>
      { state: { ..state, phase: Quarantined, }, effect: DrainRequired, }
    (Quarantined, Drained) | (Host, Evict) => release(state)
    _ => { state, effect: Rejected, }
  }
}

///|
fn pending(phase : Phase) -> Bool {
  match phase {
    Spilling | Restoring | Quarantined => true
    _ => false
  }
}
