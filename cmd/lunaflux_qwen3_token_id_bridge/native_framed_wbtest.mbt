///|
test "Qwen benchmark bridge uses the production framed receipt deadline" {
  assert_eq(bridge_request_deadline_millis(), 60000UL)
}

///|
test "Qwen benchmark bridge normalizes greedy seed to the internal contract" {
  let seed = try! @inference.SamplingSeed::new(bridge_internal_greedy_seed())
  assert_eq(seed.value(), 1UL)
}

///|
test "Qwen bridge frame decodes under the production runtime limits" {
  let request = try! @bridge.parse_qwen3_token_id_request(
    (
      #|{"schema":"lunaflux.benchmark-token-ids-request.v1","model":"Qwen3-0.6B","input_token_ids":[151644,872,198,840,20772,304,825,2805,11652,3170,72349,44378,12850,13,151645,198,151644,77091,198,151667,271,151668,271],"max_output_tokens":2,"sampling":{"mode":"greedy","temperature":0,"top_p":1,"seed":0,"ignore_eos":true},"stream":true}
    ),
    4096,
    256,
    151935,
  )
  let bridge_limits = try! bridge_limits_for_bounds(4096, 256, 151935, 40960)
  let encoded = try! bridge_request_frame_for_identity(
    "f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b", "7d13ab26706d8ca000d493ad05844489f1c0a143903838456db51c7a2e6abca7",
    request, 1UL, bridge_limits,
  )
  let runtime_limits = try! @inference.InferenceLimits::new(
    max_text_bytes=65536,
    max_input_tokens=4096,
    max_new_tokens=256,
    max_context_tokens=40960,
    max_token_id=151935,
    max_stop_token_ids=16,
    max_stop_strings=16,
    max_stop_string_bytes=256,
    max_trace_bytes=128,
    max_cache_scope_bytes=64,
    max_decoded_delta_bytes=4099,
    max_deadline_millis=60000UL,
    max_top_k=151936,
    max_temperature=2.0,
  )
  let framed = try! @framed_wire.FramedWireLimits::new(
    runtime_limits,
    max_frame_bytes=1048576,
  )
  let storage = FixedArray::make(encoded.length(), b'\x00')
  for index in 0..<encoded.length() {
    storage[index] = encoded[index]
  }
  let decoded = try! @framed_wire.RequestFrameBuffer::new(framed)
    .load(storage, encoded.length())
    .request()
  assert_eq(decoded.sampling_seed().value(), 1UL)
  assert_eq(decoded.deadline().milliseconds(), 60000UL)
  assert_eq(decoded.request_id().value(), 1UL)
}
