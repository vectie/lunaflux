///|
test "streaming frames retain graph predecessor and reject unmatched rank replies" {
  let (group, _, bindings) = ready_wire_transcript()
  let limits = fixture_wire_limits()
  let wire = FixedArray::make(limits.max_frame_bytes(), b'\x00')
  let payload = FixedArray::make(64, b'\x00')
  for sequence in 1UL..<4UL {
    for rank in 0..<2 {
      let request = @rank_group_wire.encode_rank_group_wire_payload_into(
        wire,
        limits,
        bindings[rank],
        StreamingRequest,
        ParentToChild,
        None,
        sequence,
        payload,
        64,
      )
      ignore(group.accept_bytes(wire, request, limits))
      let response = @rank_group_wire.encode_rank_group_wire_payload_into(
        wire,
        limits,
        bindings[rank],
        StreamingReply,
        ChildToParent,
        None,
        sequence,
        payload,
        64,
      )
      ignore(group.accept_bytes(wire, response, limits))
    }
  }
  assert_eq(group.previous_sequence_value(), 0UL)
  assert_eq(group.lifecycle(), ReadyGroup)
  let unsolicited = @rank_group_wire.encode_rank_group_wire_payload_into(
    wire,
    limits,
    bindings[1],
    StreamingReply,
    ChildToParent,
    None,
    4UL,
    payload,
    64,
  )
  try group.accept_bytes(wire, unsolicited, limits) catch {
    _ => ()
  } noraise {
    _ => fail("unsolicited reply accepted")
  }
  assert_eq(group.lifecycle(), Faulted)
}

///|
test "rank streaming request fences graph submission and binds response sequence" {
  let (rank, _, limits, binding, wire) = ready_local_rank_fixture()
  let payload = FixedArray::make(64, b'\x00')
  let request = @rank_group_wire.encode_rank_group_wire_payload_into(
    wire,
    limits,
    binding,
    StreamingRequest,
    ParentToChild,
    None,
    8UL,
    payload,
    64,
  )
  ignore(rank.accept_parent(wire, request, limits))
  let wrong = @rank_group_wire.encode_rank_group_wire_payload_into(
    wire,
    limits,
    binding,
    StreamingReply,
    ChildToParent,
    None,
    9UL,
    payload,
    64,
  )
  try rank.accept_child(wire, wrong, limits) catch {
    _ => ()
  } noraise {
    _ => fail("wrong sequence accepted")
  }
  let (other, _, limits, binding, wire) = ready_local_rank_fixture()
  let request = @rank_group_wire.encode_rank_group_wire_payload_into(
    wire,
    limits,
    binding,
    StreamingRequest,
    ParentToChild,
    None,
    1UL,
    payload,
    64,
  )
  ignore(other.accept_parent(wire, request, limits))
  let submit = @rank_group_wire.encode_rank_group_wire_payload_into(
    wire,
    limits,
    binding,
    Submit,
    ParentToChild,
    Some(A),
    1UL,
    payload,
    64,
  )
  try other.accept_parent(wire, submit, limits) catch {
    _ => ()
  } noraise {
    _ => fail("graph interleaved with transfer")
  }
}
