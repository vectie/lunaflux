///|
fn complete_wire_plan(
  transcript : @rank_group_wire.RankGroupWireTranscript,
  sender : @rank_group_wire.RankGroupWireFrameBuffer,
  bindings : Array[@rank_group_wire.RankGroupWireBinding],
  slot : @rank_group_wire.RankGroupWireSlot,
  sequence : UInt64,
) -> Unit {
  let plan_payload = FixedArray::make(3, b'\x2a')
  for binding in bindings {
    let submit = try! sender.encode_payload(
      binding,
      Submit,
      ParentToChild,
      slot,
      sequence,
      plan_payload,
      3,
    )
    try! transcript.accept(submit)
  }
  accept_wire_control(
    transcript,
    sender,
    bindings[0],
    PollComplete,
    ParentToChild,
    Some(slot),
    sequence,
  )
  accept_wire_control(
    transcript,
    sender,
    bindings[0],
    Pending,
    ChildToParent,
    Some(slot),
    sequence,
  )
  accept_wire_control(
    transcript,
    sender,
    bindings[0],
    PollComplete,
    ParentToChild,
    Some(slot),
    sequence,
  )
  let completion = FixedArray::make(2, b'\x17')
  let leader = try! sender.encode_payload(
    bindings[0],
    Complete,
    ChildToParent,
    slot,
    sequence,
    completion,
    2,
  )
  try! transcript.accept(leader)
  accept_wire_control(
    transcript,
    sender,
    bindings[1],
    PollComplete,
    ParentToChild,
    Some(slot),
    sequence,
  )
  accept_wire_control(
    transcript,
    sender,
    bindings[1],
    Complete,
    ChildToParent,
    Some(slot),
    sequence,
  )
}

///|
test "rank group wire orders configure ready A B drain and close" {
  let (transcript, sender, bindings) = ready_wire_transcript()
  assert_eq(transcript.lifecycle(), ReadyGroup)
  complete_wire_plan(transcript, sender, bindings, A, 1UL)
  assert_eq(transcript.previous_sequence_value(), 1UL)
  complete_wire_plan(transcript, sender, bindings, B, 2UL)
  assert_eq(transcript.previous_sequence_value(), 2UL)
  for binding in bindings {
    accept_wire_control(
      transcript,
      sender,
      binding,
      Drain,
      ParentToChild,
      None,
      0UL,
    )
  }
  for binding in bindings {
    accept_wire_control(
      transcript,
      sender,
      binding,
      Drained,
      ChildToParent,
      None,
      0UL,
    )
  }
  for binding in bindings {
    accept_wire_control(
      transcript,
      sender,
      binding,
      Close,
      ParentToChild,
      None,
      0UL,
    )
  }
  for binding in bindings {
    accept_wire_control(
      transcript,
      sender,
      binding,
      Closed,
      ChildToParent,
      None,
      0UL,
    )
  }
  assert_eq(transcript.lifecycle(), ClosedGroup)
}

///|
test "rank group wire aborts while a completion poll is pending" {
  let (transcript, sender, bindings) = ready_wire_transcript()
  let payload = FixedArray::make(1, b'\x01')
  for binding in bindings {
    let submit = try! sender.encode_payload(
      binding,
      Submit,
      ParentToChild,
      A,
      1UL,
      payload,
      1,
    )
    try! transcript.accept(submit)
  }
  accept_wire_control(
    transcript,
    sender,
    bindings[0],
    PollComplete,
    ParentToChild,
    Some(A),
    1UL,
  )
  for binding in bindings {
    accept_wire_control(
      transcript,
      sender,
      binding,
      Abort,
      ParentToChild,
      Some(A),
      1UL,
    )
  }
  for binding in bindings {
    accept_wire_control(
      transcript,
      sender,
      binding,
      Aborted,
      ChildToParent,
      Some(A),
      1UL,
    )
  }
  assert_eq(transcript.lifecycle(), AbortedPlan)
  assert_eq(transcript.previous_sequence_value(), 1UL)
  let repeated_abort = try! sender.encode_control(
    bindings[0],
    Abort,
    ParentToChild,
    None,
    0UL,
  )
  try transcript.accept(repeated_abort) catch {
    ProtocolFault(Lifecycle) => ()
    _ => fail("unexpected post-abort Abort error")
  } noraise {
    _ => fail("group transcript admitted Abort after AbortedPlan")
  }
}

///|
test "rank group wire transport loss faults immediately without an ACK state" {
  let (transcript, sender, bindings) = ready_wire_transcript()
  let payload = FixedArray::make(1, b'\x01')
  for binding in bindings {
    let submit = try! sender.encode_payload(
      binding,
      Submit,
      ParentToChild,
      A,
      1UL,
      payload,
      1,
    )
    try! transcript.accept(submit)
  }
  accept_wire_control(
    transcript,
    sender,
    bindings[1],
    PollComplete,
    ParentToChild,
    Some(A),
    1UL,
  )
  transcript.abandon_on_transport_loss()
  assert_eq(transcript.lifecycle(), Faulted)
  assert_eq(transcript.failure_rule(), Some(TransportLost))
  transcript.abandon_on_transport_loss()
  assert_eq(transcript.failure_rule(), Some(TransportLost))
}
