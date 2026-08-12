///|
test "protocol version is stable" {
  inspect(@contracts.protocol_version(), content="lunaflux.inference.v1alpha1")
}

///|
test "request state machine accepts only declared edges" {
  assert_true(@contracts.RequestState::Waiting.can_transition_to(Prefill))
  assert_true(@contracts.RequestState::Decode.can_transition_to(Decode))
  assert_false(@contracts.RequestState::Finished.can_transition_to(Decode))
  assert_false(@contracts.RequestState::Waiting.can_transition_to(Finished))
}

///|
fn expected_transition(
  current : @contracts.RequestState,
  next : @contracts.RequestState,
) -> Bool {
  match (current, next) {
    (Admitted, Tokenizing | Cancelled | Failed) => true
    (Tokenizing, Waiting | Cancelled | Failed) => true
    (Waiting, Prefill | Cancelled | Failed) => true
    (Prefill, Prefill | Decode | Finishing | Cancelled | Failed) => true
    (Decode, Decode | Finishing | Cancelled | Failed) => true
    (Finishing, Finished | Cancelled | Failed) => true
    _ => false
  }
}

///|
test "request transition table is exhaustive and terminal states stay terminal" {
  let states : Array[@contracts.RequestState] = [
    Admitted,
    Tokenizing,
    Waiting,
    Prefill,
    Decode,
    Finishing,
    Cancelled,
    Finished,
    Failed,
  ]
  for current in states {
    for next in states {
      assert_eq(
        current.can_transition_to(next),
        expected_transition(current, next),
      )
    }
  }
}
