///|
test "exhausted generations retire rather than alias a stale entry" {
  let old : State = {
    phase: Host,
    generation: 2147483647,
    epoch: 1,
    cancelled: false,
  }
  let next = transition(old, Evict)
  assert_true(next.effect is Released)
  assert_true(next.state.phase is Retired)
  assert_true(transition(next.state, BeginSpill).effect is Rejected)
}

///|
test "transfer epoch exhaustion refuses restore but permits eviction" {
  let old : State = {
    phase: Host,
    generation: 3,
    epoch: 2147483647,
    cancelled: false,
  }
  assert_true(transition(old, BeginRestore).effect is Rejected)
  let next = transition(old, Evict)
  assert_true(next.state.phase is Free)
  assert_eq(next.state.generation, 4)
  assert_eq(next.state.epoch, 0)
}

///|
test "only completion or drain can release a pending transfer" {
  for phase in [Spilling, Restoring, Quarantined] {
    for cancelled in [false, true] {
      let state : State = { phase, generation: 1, epoch: 8, cancelled, }
      for
        event in [
          BeginSpill,
          BeginRestore,
          Cancel,
          Complete,
          Fail,
          Drained,
          Evict,
        ] {
        let result = transition(state, event)
        if !pending(result.state.phase) {
          assert_true(event is Complete || event is Drained)
        }
        if result.effect is Rejected {
          assert_eq(result.state.generation, state.generation)
          assert_eq(result.state.epoch, state.epoch)
          assert_eq(result.state.cancelled, state.cancelled)
        }
      }
    }
  }
}
