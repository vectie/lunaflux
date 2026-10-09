///|
test "audio composition rejects shared and duplicate owners without partial claims" {
  let a : AudioDecoderLease = { held: false, }
  let b : AudioDecoderLease = { held: false, }
  let c : AudioDecoderLease = { held: false, }
  try audio_composition_preflight([a, b, a]) catch {
    _ => ()
  } noraise {
    _ => fail("duplicate composition owner accepted")
  }
  assert_false(a.held)
  assert_false(b.held)
  audio_composition_preflight([a, b])
  // Successful resource allocation happens before this claim.
  audio_composition_claim([a, b])
  try audio_composition_preflight([c, b]) catch {
    _ => ()
  } noraise {
    _ => fail("already borrowed component accepted")
  }
  assert_false(c.held)
  assert_true(a.held && b.held)
  // Failed storage release does not call release; retry retains both claims.
  try a.require_idle() catch {
    _ => ()
  } noraise {
    _ => fail("component close accepted while composed")
  }
  audio_composition_release([a, b])
  audio_composition_preflight([c, b])
  audio_composition_claim([c, b])
  assert_true(c.held && b.held)
  assert_false(a.held)
}
