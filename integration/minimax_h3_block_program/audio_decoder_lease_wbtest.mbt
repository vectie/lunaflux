///|
test "decoder workspace stays exclusive until request release" {
  let lease : AudioDecoderLease = { held: false, }
  lease.require_idle()
  lease.acquire()
  try lease.acquire() catch {
    _ => ()
  } noraise {
    _ => fail("overlapping request acquired shared scratch")
  }
  try lease.require_idle() catch {
    _ => ()
  } noraise {
    _ => fail("active decoder permitted close")
  }
  assert_true(lease.held)
  lease.held = false
  lease.require_idle()
  lease.acquire()
  assert_true(lease.held)
}
