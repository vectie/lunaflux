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
    device_ordinal=3,
    model_generation_value=9UL,
    worker_limits=wire_protocol_limits(),
    inference_limits=wire_inference_limits(),
    has_predecessor=true,
    predecessor_value=7UL,
  )
}

///|
test "startup frame round trips exact contract and kind" {
  let sender = @worker_wire.StartupFrameBuffer::new()
  let expected = startup_contract()
  let configured = sender.encode(Configure, expected)
  let transport = FixedArray::make(sender.capacity(), b'\x00')
  configured.copy_to(transport, destination_offset=0)
  let receiver = @worker_wire.StartupFrameBuffer::new()
  let received = receiver.load(transport, configured.length())
  assert_eq(received.kind(), Configure)
  assert_eq(received.contract(), expected)
  assert_eq(received.contract().device_ordinal(), 3)
  let ready = sender.encode(Ready, received.contract())
  assert_eq(ready.kind(), Ready)
  assert_eq(ready.contract(), expected)
}

///|
test "startup frame corruption and stale epochs fail closed" {
  let owner = @worker_wire.StartupFrameBuffer::new()
  let first = owner.encode(Configure, startup_contract())
  let transport = FixedArray::make(owner.capacity(), b'\x00')
  first.copy_to(transport, destination_offset=0)
  transport[40] = transport[40] ^ b'\x01'
  try owner.load(transport, transport.length()) catch {
    InvalidFrame(Checksum) => ()
    _ => fail("unexpected startup corruption error")
  } noraise {
    _ => fail("corrupt startup frame was accepted")
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
test "startup predecessor identity is canonical" {
  let base = startup_contract()
  for candidate in [(true, 0UL), (true, 0xffffffffffffffffUL), (false, 1UL)] {
    let (has_predecessor, predecessor) = candidate
    try
      @worker_wire.WorkerStartupContract::new(
        identity=base.identity(),
        bootstrap_digest=base.bootstrap_digest(),
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
