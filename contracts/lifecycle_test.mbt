///|
test "protocol version is stable" {
  inspect(@contracts.protocol_version(), content="lunaflux.inference.v1alpha1")
}

///|
test "request state machine accepts only declared edges" {
  assert_true(
    @contracts.RequestState::Waiting.can_transition_to(
      @contracts.RequestState::Prefill,
    ),
  )
  assert_true(
    @contracts.RequestState::Decode.can_transition_to(
      @contracts.RequestState::Decode,
    ),
  )
  assert_false(
    @contracts.RequestState::Finished.can_transition_to(
      @contracts.RequestState::Decode,
    ),
  )
  assert_false(
    @contracts.RequestState::Waiting.can_transition_to(
      @contracts.RequestState::Finished,
    ),
  )
}
