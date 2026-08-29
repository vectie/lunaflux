///|
test "plan frame round trips exact scalar tables and rows" {
  let sender = @worker_wire.PlanFrameBuffer::new(
    wire_protocol_limits(),
    wire_inference_limits(),
  )
  let encoded = sender.encode_plan(submitted_wire_fixture())
  assert_eq(encoded.length(), 284)
  assert_eq(encoded.sequence_value(), 1UL)
  assert_eq(encoded.model_generation_value(), 9UL)
  assert_eq(encoded.token_budget(), 3)
  assert_eq(encoded.token_count(), 3)
  assert_eq(encoded.page_count(), 2)
  assert_eq(encoded.capability_count(), 2)
  assert_eq(encoded.prefill_count(), 1)
  assert_eq(encoded.decode_count(), 1)
  assert_eq(encoded.token_at(0), 10)
  assert_eq(encoded.token_at(1), 11)
  assert_eq(encoded.token_at(2), 12)
  assert_eq(encoded.page_index_at(0), 0)
  assert_eq(encoded.page_generation_at(0), 1)
  assert_eq(encoded.page_index_at(1), 1)
  assert_eq(
    encoded.capability_value_at(0),
    @plan.KernelCapabilityId::positioned_rotary_embedding().as_int(),
  )
  assert_eq(
    encoded.capability_value_at(1),
    @plan.KernelCapabilityId::paged_causal_attention().as_int(),
  )
  let prefill = encoded.prefill_row(0)
  assert_eq(prefill.request_id_value(), 7UL)
  assert_eq(prefill.request_generation_value(), 0UL)
  assert_eq(prefill.sequence_token_start(), 3)
  assert_eq(prefill.token_start(), 0)
  assert_eq(prefill.token_length(), 2)
  assert_eq(prefill.page_start(), 0)
  assert_eq(prefill.page_length(), 1)
  assert_eq(prefill.capability_start(), 0)
  assert_eq(prefill.capability_length(), 1)
  assert_true(prefill.produces_token())
  assert_eq(prefill.sampling_mode_value(), 1)
  assert_eq(prefill.sampling_temperature(), 0.75)
  assert_true(prefill.sampling_has_top_k())
  assert_eq(prefill.sampling_top_k_value(), 16)
  assert_true(prefill.sampling_has_top_p())
  assert_eq(prefill.sampling_top_p_value(), 0.9)
  assert_eq(prefill.sampling_seed_value(), 0x1234UL)
  assert_eq(prefill.sample_index(), 5)
  assert_eq(prefill.completion_slot(), 0)
  let decode = encoded.decode_row(0)
  assert_eq(decode.request_id_value(), 8UL)
  assert_eq(decode.request_generation_value(), 3UL)
  assert_eq(decode.sequence_token_position(), 6)
  assert_eq(decode.token_start(), 2)
  assert_eq(decode.page_start(), 1)
  assert_eq(decode.page_length(), 1)
  assert_eq(decode.capability_start(), 1)
  assert_eq(decode.capability_length(), 1)
  assert_eq(decode.sampling_mode_value(), 0)
  assert_false(decode.sampling_has_top_k())
  assert_eq(decode.sampling_top_k_value(), 0)
  assert_false(decode.sampling_has_top_p())
  assert_eq(decode.sampling_top_p_value(), 0.0)
  assert_eq(decode.sampling_seed_value(), 0x5678UL)
  assert_eq(decode.sample_index(), 6)
  assert_eq(decode.completion_slot(), 1)
  let transport = FixedArray::make(encoded.length(), b'\x00')
  encoded.copy_to(transport, destination_offset=0)
  let receiver = @worker_wire.PlanFrameBuffer::new(
    wire_protocol_limits(),
    wire_inference_limits(),
  )
  let received = receiver.load(transport, transport.length())
  assert_eq(received.sequence_value(), encoded.sequence_value())
  assert_eq(received.decode_row(0).sampling_seed_value(), 0x5678UL)
}

///|
test "re-encoding invalidates prior authenticated views" {
  let owner = @worker_wire.PlanFrameBuffer::new(
    wire_protocol_limits(),
    wire_inference_limits(),
  )
  let first = owner.encode_plan(submitted_wire_fixture())
  ignore(owner.encode_plan(submitted_wire_fixture()))
  try first.length() catch {
    StaleFrame => ()
    _ => fail("expected stale frame")
  } noraise {
    _ => fail("expected stale frame rejection")
  }
}

///|
test "current plan view rederives only the retained encoded epoch" {
  let owner = @worker_wire.PlanFrameBuffer::new(
    wire_protocol_limits(),
    wire_inference_limits(),
  )
  try owner.current() catch {
    StaleFrame => ()
    _ => fail("unexpected empty current frame error")
  } noraise {
    _ => fail("empty plan buffer exposed a current frame")
  }
  let encoded = owner.encode_plan(submitted_wire_fixture())
  let current = owner.current()
  assert_eq(current.epoch_value(), encoded.epoch_value())
  assert_eq(current.length(), encoded.length())
  assert_eq(current.sequence_value(), encoded.sequence_value())
}

///|
test "invalid transport preserves prior received frame" {
  let sender = @worker_wire.PlanFrameBuffer::new(
    wire_protocol_limits(),
    wire_inference_limits(),
  )
  let encoded = sender.encode_plan(submitted_wire_fixture())
  let transport = FixedArray::make(encoded.length(), b'\x00')
  encoded.copy_to(transport, destination_offset=0)
  let receiver = @worker_wire.PlanFrameBuffer::new(
    wire_protocol_limits(),
    wire_inference_limits(),
  )
  let prior = receiver.load(transport, transport.length())
  transport[0] = transport[0] ^ b'\x01'
  try receiver.load(transport, transport.length()) catch {
    InvalidFrame(Magic) => ()
    _ => fail("expected magic rejection")
  } noraise {
    _ => fail("expected invalid frame")
  }
  assert_eq(prior.sequence_value(), 1UL)
  assert_eq(prior.token_at(2), 12)
}

///|
test "copy and load reject undersized storage" {
  let owner = @worker_wire.PlanFrameBuffer::new(
    wire_protocol_limits(),
    wire_inference_limits(),
  )
  let frame = owner.encode_plan(submitted_wire_fixture())
  try frame.copy_to(FixedArray::make(8, b'\x00'), destination_offset=0) catch {
    DestinationTooSmall => ()
    _ => fail("expected destination rejection")
  } noraise {
    _ => fail("expected destination rejection")
  }
  let receiver = @worker_wire.PlanFrameBuffer::new(
    wire_protocol_limits(),
    wire_inference_limits(),
  )
  try receiver.load(FixedArray::make(8, b'\x00'), 8) catch {
    SourceTooSmall => ()
    _ => fail("expected source rejection")
  } noraise {
    _ => fail("expected source rejection")
  }
}

///|
test "canonical encoding is byte-for-byte deterministic" {
  let left_owner = @worker_wire.PlanFrameBuffer::new(
    wire_protocol_limits(),
    wire_inference_limits(),
  )
  let right_owner = @worker_wire.PlanFrameBuffer::new(
    wire_protocol_limits(),
    wire_inference_limits(),
  )
  let left = left_owner.encode_plan(submitted_wire_fixture())
  let right = right_owner.encode_plan(submitted_wire_fixture())
  assert_eq(left.length(), right.length())
  let left_bytes = FixedArray::make(left.length(), b'\x00')
  let right_bytes = FixedArray::make(right.length(), b'\x00')
  left.copy_to(left_bytes, destination_offset=0)
  right.copy_to(right_bytes, destination_offset=0)
  for index in 0..<left.length() {
    assert_eq(left_bytes[index], right_bytes[index])
  }
}

///|
test "transport-owned plan frame encodes at its immutable offset" {
  let limits = wire_protocol_limits()
  let offset = 17
  let storage = FixedArray::make(
    offset + plan_frame_byte_capacity(limits),
    b'\x5a',
  )
  let owner = PlanFrameBuffer::new_transport_owned(
    limits,
    wire_inference_limits(),
    storage,
    frame_offset=offset,
  )
  let frame = owner.encode_plan(submitted_wire_fixture())
  assert_eq(frame.transport_offset(), offset)
  for index in 0..<offset {
    assert_eq(storage[index], b'\x5a')
  }
  let copied = FixedArray::make(frame.length(), b'\x00')
  frame.copy_to(copied, destination_offset=0)
  let receiver = PlanFrameBuffer::new(limits, wire_inference_limits())
  assert_eq(receiver.load(copied, copied.length()).sequence_value(), 1UL)
}
