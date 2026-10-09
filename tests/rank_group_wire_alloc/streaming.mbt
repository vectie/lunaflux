///|
fn verify_streaming_wire() -> Unit {
  let limits = try! @rank_group_wire.RankGroupWireLimits::new(
    max_payload_bytes=64,
  )
  let bindings = FixedArray::makei(2, rank => binding(rank))
  let transcript = try! @rank_group_wire.RankGroupWireTranscript::new(
    bindings,
    previous_sequence_value=0UL,
  )
  let sender = @rank_group_wire.RankGroupWireFrameBuffer::new(limits)
  let wire = FixedArray::make(limits.max_frame_bytes(), b'\x00')
  let payload = FixedArray::make(64, b'\x01')
  start_ready(transcript, sender, bindings)
  alloc_probe_begin()
  for sequence in 1UL..<1001UL {
    for rank in 0..<2 {
      let length = try! @rank_group_wire.encode_rank_group_wire_payload_into(
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
      ignore(try! transcript.accept_bytes(wire, length, limits))
      let length = try! @rank_group_wire.encode_rank_group_wire_payload_into(
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
      ignore(try! transcript.accept_bytes(wire, length, limits))
    }
  }
  if alloc_probe_check(alloc_probe_end(), 1) != 1 ||
    transcript.previous_sequence_value() != 0UL {
    alloc_probe_abort()
  }
}
