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
