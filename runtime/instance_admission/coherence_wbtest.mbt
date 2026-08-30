///|
test "pure instance join rejects descriptor tokenizer layout and capacity substitutions" {
  let digest = try! @descriptor_file.RuntimeDescriptorDigest::from_sha256(
    "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
  )
  try! require_runtime_digest(
    "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef", digest,
  )
  let tokenizer = try! @tokenizer.TokenizerDigest::parse(
    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
  )
  try! require_tokenizer_identity(
    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", tokenizer,
  )
  let hostile : Array[() -> Unit raise RuntimeInstanceAdmissionError] = [
    () => {
      require_runtime_digest(
        "1123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef", digest,
      )
    },
    () => require_layout_geometry(16, 128, 1, 32, 128, 1),
    () => {
      require_tokenizer_identity(
        "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb", tokenizer,
      )
    },
    () => require_tokenizer_envelope(32001, 32000, 4096, 4096, 128, 128, 8, 8),
    () => require_blueprint_capacity(8, 7),
    () => require_service_frame_bytes(4096, 8192),
    () => require_preparation_lanes(7, 8),
  ]
  for index, check in hostile {
    try check() catch {
      Rejected(RuntimeDigest) if index == 0 => ()
      Rejected(Layout) if index == 1 => ()
      Rejected(TokenizerBinding) if index == 2 => ()
      Rejected(TokenizerEnvelope) if index == 3 => ()
      Rejected(Capacity) if index == 4 => ()
      Rejected(FramedWire) if index == 5 => ()
      Rejected(Preparation) if index == 6 => ()
      _ => fail("unexpected instance coherence substitution failure")
    } noraise {
      _ => fail("instance coherence substitution was admitted")
    }
  }
}

///|
test "tokenizer vocabulary may occupy a padded model vocabulary prefix" {
  try! require_tokenizer_envelope(
    151669, 151936, 65536, 65536, 65536, 4096, 1048576, 4096,
  )
}

///|
test "worker process capacity uses exact public wire formulas" {
  let worker = try! @worker_protocol.WorkerProtocolLimits::new(
    max_prefill_rows=2,
    max_decode_rows=2,
    max_plan_rows=2,
    max_plan_tokens=64,
    max_plan_pages=16,
    max_capabilities=64,
    max_completion_slots=2,
    max_sequence_tokens=128,
    max_token_id=255,
  )
  let expected = Int::max(
    @worker_wire.bootstrap_source_capacity(),
    Int::max(
      try! @worker_wire.plan_frame_byte_capacity(worker),
      try! @worker_wire.completion_frame_byte_capacity(worker),
    ),
  )
  assert_eq(try! required_worker_frame_capacity(worker), expected)
}
