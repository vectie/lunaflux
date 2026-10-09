///|
fn verify_streaming_control(
  owner : @rank_child_control.RankChildControl,
  limits : @rank_group_wire.RankGroupWireLimits,
  binding : @rank_group_wire.RankGroupWireBinding,
  parent_wire : FixedArray[Byte],
  child_wire : FixedArray[Byte],
) -> Unit {
  let payload = FixedArray::make(64, b'\x73')
  alloc_probe_begin()
  for sequence in 1UL..<1001UL {
    let length = try! @rank_group_wire.encode_rank_group_wire_payload_into(
      parent_wire,
      limits,
      binding,
      StreamingRequest,
      ParentToChild,
      None,
      sequence,
      payload,
      64,
    )
    send_parent_frame(parent_wire, length)
    let inbound = receive_inbound(owner)
    if (try! inbound.kind()) != StreamingRequest ||
      (try! inbound.sequence_value()) != sequence {
      alloc_probe_abort()
    }
    try! owner.begin_streaming_reply(inbound, payload)
    finish_send(owner)
    receive_child_frame(
      child_wire,
      limits,
      binding,
      StreamingReply,
      None,
      sequence,
      payload,
      64,
    )
  }
  if alloc_probe_check(alloc_probe_end(), 1) != 1 {
    alloc_probe_abort()
  }
}
