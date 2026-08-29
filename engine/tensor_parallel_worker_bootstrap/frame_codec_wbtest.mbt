///|
test "fixed rank envelope codec round trips every bounded identity" {
  let group = bootstrap_test_group()
  let envelope = group.envelopes[1]
  let sender = TensorParallelRankEnvelopeFrameBuffer::new()
  let frame = sender.encode(envelope)
  assert_eq(frame.length(), 752)
  let wire = FixedArray::make(752, b'\x00')
  frame.copy_to(wire, destination_offset=0)
  let receiver = TensorParallelRankEnvelopeFrameBuffer::new()
  let decoded = receiver.load(wire, 752).envelope()
  assert_eq(decoded.identity, envelope.identity)
  assert_eq(decoded.model_generation_value, 7UL)
  assert_eq(decoded.group_generation_value, 9UL)
  assert_eq(decoded.rank, 1)
  assert_eq(decoded.world_size, 2)
  assert_eq(decoded.device_ordinal, 5)
  assert_eq(decoded.collective_runtime_version_code, 21403)
  assert_eq(decoded.group_digest, envelope.group_digest)
  assert_eq(decoded.topology_digest, envelope.topology_digest)
  assert_eq(decoded.worker_digest, envelope.worker_digest)
  assert_eq(decoded.device_plan_digest, envelope.device_plan_digest)
  assert_eq(
    decoded.execution_plan_digest_sha256,
    envelope.execution_plan_digest_sha256,
  )
  assert_eq(decoded.collective_digest, envelope.collective_digest)
  assert_eq(decoded.kv_digest, envelope.kv_digest)
  assert_true(same_rendezvous_id(decoded.rendezvous_id, envelope.rendezvous_id))
}

///|
test "frame rejects corruption oversized lengths and stale epochs" {
  let group = bootstrap_test_group()
  let sender = TensorParallelRankEnvelopeFrameBuffer::new()
  let first = sender.encode(group.envelopes[0])
  let wire = FixedArray::make(752, b'\x00')
  first.copy_to(wire, destination_offset=0)
  let receiver = TensorParallelRankEnvelopeFrameBuffer::new()
  try receiver.load(wire, 753) catch {
    Rejected(rule=FrameLength, rank=-1) => ()
    _ => fail("unexpected oversized-frame rejection")
  } noraise {
    _ => fail("oversized frame was admitted")
  }
  wire[304] = wire[304] ^ b'\x01'
  try receiver.load(wire, 752) catch {
    Rejected(rule=FrameChecksum, rank=-1) => ()
    _ => fail("unexpected corruption rejection")
  } noraise {
    _ => fail("corrupt frame was admitted")
  }
  first.copy_to(wire, destination_offset=0)
  wire[432] = b'g'
  frame_write_u32(
    wire,
    RANK_ENVELOPE_CHECKSUM_OFFSET,
    rank_envelope_checksum(wire),
  )
  try receiver.load(wire, 752) catch {
    Rejected(rule=Digest, rank=-1) => ()
    _ => fail("unexpected execution-plan digest rejection")
  } noraise {
    _ => fail("noncanonical execution-plan digest was admitted")
  }
  ignore(sender.encode(group.envelopes[1]))
  try first.length() catch {
    StaleFrame => ()
    _ => fail("unexpected stale-frame rejection")
  } noraise {
    _ => fail("stale frame remained live")
  }
}

///|
test "frame destination bound fails before copying" {
  let group = bootstrap_test_group()
  let sender = TensorParallelRankEnvelopeFrameBuffer::new()
  let frame = sender.encode(group.envelopes[0])
  let short = FixedArray::make(751, b'\x00')
  try frame.copy_to(short, destination_offset=0) catch {
    DestinationTooSmall => ()
    _ => fail("unexpected destination bound rejection")
  } noraise {
    _ => fail("short destination was accepted")
  }
}
