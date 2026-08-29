///|
test "optional promotion verifier distinguishes missing and invalid FD 7" {
  ignore(fixture_clear_fixed())
  assert_true(
    @manifest.prepare_optional_inherited_luna_startup_verifier_key_v1() is None,
  )
  let invalid = fixture_install(1)
  defer ignore(fixture_close(invalid))
  try
    @manifest.prepare_optional_inherited_luna_startup_verifier_key_v1()
  catch {
    Invalid(ExternalApproval) => ()
    _ => abort("wrong descriptor returned the wrong verifier refusal")
  } noraise {
    _ => abort("wrong verifier descriptor was admitted")
  }
  assert_true(fixture_fixed_closed())
}

///|
test "fragmented exact verifier frame becomes one opaque handoff" {
  let peer = fixture_install(0)
  defer ignore(fixture_close(peer))
  guard @manifest.prepare_optional_inherited_luna_startup_verifier_key_v1()
    is Some(owner) else {
    abort("installed verifier was unavailable")
  }
  assert_true(fixture_fixed_closed())
  let frame = verifier_frame()
  assert_eq(fixture_send(peer, frame, 0, 7), 7)
  assert_eq(owner.progress(), LunaStartupVerifierKeyPending)
  assert_eq(fixture_send(peer, frame, 7, 19), 19)
  assert_eq(owner.progress(), LunaStartupVerifierKeyPending)
  assert_eq(
    fixture_send(peer, frame, 26, frame.length() - 26),
    frame.length() - 26,
  )
  assert_eq(owner.progress(), LunaStartupVerifierKeyPending)
  assert_eq(fixture_close_write(peer), 0)
  assert_eq(progress_until_terminal(owner), LunaStartupVerifierKeyReady)
  let handoff = owner.take_handoff()
  handoff.close()
  handoff.close()
  try owner.take_handoff() catch {
    Invalid(ExternalApproval) => ()
    _ => abort("replayed frame returned the wrong verifier refusal")
  } noraise {
    _ => abort("one verifier frame minted two handoffs")
  }
}

///|
test "verifier framing rejects malformed trailing and truncated input" {
  for mode in 0..<3 {
    let peer = fixture_install(0)
    guard @manifest.prepare_optional_inherited_luna_startup_verifier_key_v1()
      is Some(owner) else {
      abort("installed hostile verifier was unavailable")
    }
    let frame = verifier_frame()
    if mode == 0 {
      frame[0] = b'X'
      ignore(fixture_send(peer, frame, 0, frame.length()))
    } else if mode == 1 {
      ignore(fixture_send(peer, frame, 0, frame.length()))
      ignore(fixture_send(peer, frame, 0, 1))
    } else {
      ignore(fixture_send(peer, frame, 0, 12))
    }
    ignore(fixture_close_write(peer))
    assert_eq(
      progress_until_terminal(owner),
      LunaStartupVerifierKeyFailedProgress,
    )
    try owner.take_handoff() catch {
      Invalid(ExternalApproval) => ()
      _ => abort("hostile verifier frame returned the wrong refusal")
    } noraise {
      _ => abort("hostile verifier frame reached ready")
    }
    ignore(fixture_close(peer))
  }
}

///|
test "verifier handoff rejects a hostile policy identifier once" {
  let peer = fixture_install(0)
  defer ignore(fixture_close(peer))
  guard @manifest.prepare_optional_inherited_luna_startup_verifier_key_v1()
    is Some(owner) else {
    abort("installed hostile verifier was unavailable")
  }
  let frame = verifier_frame()
  // Payload starts after the 16-byte header and its two 32-byte digests. The
  // configured key is 32 bytes, so this is the first verifier-id byte.
  frame[112] = b'/'
  assert_eq(fixture_send(peer, frame, 0, frame.length()), frame.length())
  assert_eq(fixture_close_write(peer), 0)
  assert_eq(progress_until_terminal(owner), LunaStartupVerifierKeyReady)
  try owner.take_handoff() catch {
    Invalid(ExternalApproval) => ()
    _ => abort("hostile policy identifier returned the wrong refusal")
  } noraise {
    _ => abort("hostile policy identifier minted a verifier handoff")
  }
  assert_eq(owner.progress(), LunaStartupVerifierKeyFailedProgress)
  owner.close()
}

///|
test "verifier owner close is deterministic" {
  let peer = fixture_install(0)
  defer ignore(fixture_close(peer))
  guard @manifest.prepare_optional_inherited_luna_startup_verifier_key_v1()
    is Some(owner) else {
    abort("installed verifier was unavailable")
  }
  owner.close()
  owner.close()
  assert_eq(owner.progress(), LunaStartupVerifierKeyFailedProgress)
}
