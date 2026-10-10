///|
fn startup_contract() -> @worker_wire.WorkerStartupContract raise {
  @worker_wire.WorkerStartupContract::new(
    identity=@spec.ModelIdentity::new(
      @spec.ContentDigest::from_sha256(
        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
      ),
      @spec.PlanDigest::from_sha256(
        "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789",
      ),
    ),
    bootstrap_digest=@worker_wire.WorkerBootstrapDigest::from_sha256(
      "00112233445566778899aabbccddeeff00112233445566778899aabbccddeeff",
    ),
    bootstrap_source_digest=@worker_wire.WorkerBootstrapSourceDigest::from_sha256(
      "112233445566778899aabbccddeeff00112233445566778899aabbccddeeff00",
    ),
    device_ordinal=3,
    model_generation_value=9UL,
    worker_limits=wire_protocol_limits(),
    inference_limits=wire_inference_limits(),
    has_predecessor=true,
    predecessor_value=7UL,
  )
}

///|
test "streaming startup v5 binds capacity measured shapes and worker epoch" {
  let cap = @streaming_ir.Capability::new(
    slots=4,
    host_bytes=256L,
    page_bytes=64L,
    page_count=16,
    reserve_pages=4,
    restore_ns=100L,
    recompute_ns=1000L,
    separate_host_memory=true,
    workspace_budget=1024L,
    shapes=[
      @streaming_ir.Shape::new(tokens=1, workspace_bytes=128L),
      @streaming_ir.Shape::new(tokens=4, workspace_bytes=1024L),
    ],
  )
  let expected = startup_contract().with_streaming(cap)
  let sender = @worker_wire.StartupFrameBuffer::new(streaming=true)
  let frame = sender.encode(Configure, expected)
  assert_eq(frame.length(), 1024)
  let bytes = FixedArray::make(1024, b'\x00')
  frame.copy_to(bytes, destination_offset=0)
  let receiver = @worker_wire.StartupFrameBuffer::new(streaming=true)
  let decoded = receiver.load(bytes, 1024).contract()
  assert_eq(decoded, expected)
  assert_eq(decoded.streaming().unwrap().shape_at(1).tokens(), 4)
  assert_eq(decoded.streaming_epoch(), 1UL)
  try @worker_wire.StartupFrameBuffer::new().load(bytes, 1024) catch {
    _ => ()
  } noraise {
    _ => fail("small legacy buffer accepted extension")
  }
  // Invalid workspace size is still rejected by the streaming capability
  // contract, independently of the removed checksum.
  for index in 488..<496 {
    bytes[index] = b'\xff'
  }
  try receiver.load(bytes, 1024).contract() catch {
    InvalidFrame(StartupIdentity) => ()
    _ => fail("wrong corruption error")
  } noraise {
    _ => fail("unbound policy")
  }
}

///|
test "startup frame round trips exact contract and kind" {
  let sender = @worker_wire.StartupFrameBuffer::new()
  assert_eq(sender.capacity(), 408)
  let expected = startup_contract()
  let configured = sender.encode(Configure, expected)
  let transport = FixedArray::make(sender.capacity(), b'\x00')
  configured.copy_to(transport, destination_offset=0)
  let receiver = @worker_wire.StartupFrameBuffer::new()
  let received = receiver.load(transport, configured.length())
  assert_eq(received.kind(), Configure)
  assert_eq(received.contract(), expected)
  assert_eq(
    received.contract().bootstrap_source_digest(),
    expected.bootstrap_source_digest(),
  )
  assert_eq(received.contract().device_ordinal(), 3)
  let ready = sender.encode(Ready, received.contract())
  assert_eq(ready.kind(), Ready)
  assert_eq(ready.contract(), expected)
}

///|
test "startup frame corruption and stale epochs fail closed" {
  let owner = @worker_wire.StartupFrameBuffer::new()
  let receiver = @worker_wire.StartupFrameBuffer::new()
  let first = owner.encode(Configure, startup_contract())
  let transport = FixedArray::make(owner.capacity(), b'\x00')
  first.copy_to(transport, destination_offset=0)
  transport[40] = b'g'
  try receiver.load(transport, transport.length()).contract() catch {
    InvalidFrame(StartupIdentity) => ()
    _ => fail("unexpected startup corruption error")
  } noraise {
    _ => fail("corrupt startup frame was accepted")
  }
  first.copy_to(transport, destination_offset=0)
  transport[4] = b'\x03'
  try receiver.load(transport, transport.length()).contract() catch {
    InvalidFrame(Version) => ()
    _ => fail("unexpected legacy startup version error")
  } noraise {
    _ => fail("legacy startup version was accepted")
  }
  first.copy_to(transport, destination_offset=0)
  transport[344] = b'g'
  try receiver.load(transport, transport.length()).contract() catch {
    InvalidFrame(StartupIdentity) => ()
    _ => fail("unexpected source-digest corruption error")
  } noraise {
    _ => fail("corrupt startup source digest was accepted")
  }
  let replacement = owner.encode(Ready, startup_contract())
  assert_eq(replacement.kind(), Ready)
  try first.kind() catch {
    StaleFrame => ()
    _ => fail("unexpected stale startup error")
  } noraise {
    _ => fail("superseded startup frame remained live")
  }
}

///|
test "startup frames ignore the unused checksum word" {
  let expected = startup_contract()
  let frame = @worker_wire.StartupFrameBuffer::new().encode(Configure, expected)
  let bytes = FixedArray::make(frame.length(), b'\x00')
  frame.copy_to(bytes, destination_offset=0)
  for index in 336..<340 {
    assert_eq(bytes[index], b'\x00')
  }
  for checksum in [b'\x00', b'\x01', b'\xff'] {
    for index in 336..<340 {
      bytes[index] = checksum
    }
    assert_eq(
      @worker_wire.StartupFrameBuffer::new()
      .load(bytes, bytes.length())
      .contract(),
      expected,
    )
  }
}

///|
test "startup source digest participates in exact contract equality" {
  let base = startup_contract()
  let alternate = @worker_wire.WorkerStartupContract::new(
    identity=base.identity(),
    bootstrap_digest=base.bootstrap_digest(),
    bootstrap_source_digest=@worker_wire.WorkerBootstrapSourceDigest::from_sha256(
      "212233445566778899aabbccddeeff00112233445566778899aabbccddeeff00",
    ),
    device_ordinal=base.device_ordinal(),
    model_generation_value=base.model_generation_value(),
    worker_limits=base.worker_limits(),
    inference_limits=base.inference_limits(),
    has_predecessor=base.has_predecessor(),
    predecessor_value=base.predecessor_value(),
  )
  assert_true(alternate != base)
  let owner = @worker_wire.StartupFrameBuffer::new()
  assert_eq(owner.encode(Configure, alternate).contract(), alternate)
}

///|
test "startup predecessor identity is canonical" {
  let base = startup_contract()
  for candidate in [(true, 0UL), (true, 0xffffffffffffffffUL), (false, 1UL)] {
    let (has_predecessor, predecessor) = candidate
    try
      @worker_wire.WorkerStartupContract::new(
        identity=base.identity(),
        bootstrap_digest=base.bootstrap_digest(),
        bootstrap_source_digest=base.bootstrap_source_digest(),
        device_ordinal=base.device_ordinal(),
        model_generation_value=base.model_generation_value(),
        worker_limits=base.worker_limits(),
        inference_limits=base.inference_limits(),
        has_predecessor~,
        predecessor_value=predecessor,
      )
    catch {
      InvalidFrame(StartupIdentity) => ()
      _ => fail("unexpected predecessor identity error")
    } noraise {
      _ => fail("noncanonical predecessor identity was accepted")
    }
  }
}

///|
test "startup bootstrap identity is exact lowercase sha256" {
  let expected = startup_contract().bootstrap_digest()
  assert_eq(
    expected.as_hex(),
    "00112233445566778899aabbccddeeff00112233445566778899aabbccddeeff",
  )
  for
    value in [
      "ABC", "00112233445566778899AABBCCDDEEFF00112233445566778899aabbccddeeff",
    ] {
    try @worker_wire.WorkerBootstrapDigest::from_sha256(value) catch {
      InvalidFrame(StartupIdentity) => ()
      _ => fail("unexpected bootstrap digest error")
    } noraise {
      _ => fail("invalid bootstrap digest was accepted")
    }
  }
}

///|
test "startup device ordinal is bounded process-visible identity" {
  let base = startup_contract()
  for ordinal in [-1, 1024] {
    try
      @worker_wire.WorkerStartupContract::new(
        identity=base.identity(),
        bootstrap_digest=base.bootstrap_digest(),
        bootstrap_source_digest=base.bootstrap_source_digest(),
        device_ordinal=ordinal,
        model_generation_value=base.model_generation_value(),
        worker_limits=base.worker_limits(),
        inference_limits=base.inference_limits(),
        has_predecessor=base.has_predecessor(),
        predecessor_value=base.predecessor_value(),
      )
    catch {
      InvalidFrame(StartupIdentity) => ()
      _ => fail("unexpected device ordinal error")
    } noraise {
      _ => fail("out-of-range device ordinal was accepted")
    }
  }
}
