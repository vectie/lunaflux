///|
test "optional credential distinguishes missing and invalid FD 6" {
  ignore(fixture_clear_fixed())
  assert_true(
    @credential.prepare_optional_inherited_inference_credential_v1() is None,
  )
  let invalid = fixture_install(1)
  defer ignore(fixture_close(invalid))
  try @credential.prepare_optional_inherited_inference_credential_v1() catch {
    InheritedInferenceCredentialFailed(CredentialDescriptorInvalid) => ()
    _ => abort("wrong descriptor returned the wrong credential rule")
  } noraise {
    _ => abort("wrong credential descriptor was admitted")
  }
  assert_true(fixture_fixed_closed())
}

///|
test "optional credential preserves runtime-owned CLOEXEC FD 6" {
  let peer = fixture_install(2)
  defer ignore(fixture_close(peer))
  assert_true(fixture_fixed_cloexec())
  assert_true(
    @credential.prepare_optional_inherited_inference_credential_v1() is None,
  )
  assert_true(fixture_fixed_cloexec())
  assert_false(fixture_fixed_closed())
  ignore(fixture_clear_fixed())
}

///|
test "fragmented credential becomes one opaque constant-bound policy" {
  let peer = fixture_install(0)
  defer ignore(fixture_close(peer))
  guard @credential.prepare_optional_inherited_inference_credential_v1()
    is Some(owner) else {
    abort("installed credential was unavailable")
  }
  assert_true(fixture_fixed_closed())
  let secret : FixedArray[Byte] = [b's', b'e', b'c', b'r', b'e', b't']
  let frame = credential_frame(secret)
  assert_eq(fixture_send(peer, frame, 0, 5), 5)
  assert_eq(owner.progress(), InheritedInferenceCredentialPending)
  assert_eq(fixture_send(peer, frame, 5, 7), 7)
  assert_eq(owner.progress(), InheritedInferenceCredentialPending)
  assert_eq(fixture_send(peer, frame, 12, secret.length()), secret.length())
  assert_eq(owner.progress(), InheritedInferenceCredentialPending)
  assert_eq(fixture_close_write(peer), 0)
  assert_eq(progress_until_terminal(owner), InheritedInferenceCredentialReady)
  let inherited = owner.take_startup_auth_policy_v1(
    maximum_credential_bytes=4096,
  )
  try ignore(inherited.take_for_policy(5)) catch {
    InheritedInferenceCredentialFailed(CredentialPolicyInvalid) => ()
    _ => abort("credential length bound returned the wrong refusal")
  } noraise {
    _ => abort("credential exceeded the authenticated policy maximum")
  }
  let policy = inherited.take_for_policy(128)
  assert_true(
    policy.verify(secret, source_offset=0, length=secret.length()).is_accepted(),
  )
  let wrong : FixedArray[Byte] = [b's', b'e', b'c', b'r', b'e', b'x']
  assert_true(
    policy.verify(wrong, source_offset=0, length=wrong.length()).is_rejected(),
  )
  inherited.close()
  inherited.close()
  try policy.verify(secret, source_offset=0, length=secret.length()) catch {
    LunaApiAuthFailed(rule=LunaApiAuthPolicy, issue=LunaApiAuthInvalid) => ()
    _ => abort("closed inherited policy returned the wrong refusal")
  } noraise {
    _ => abort("stale policy alias survived inherited close")
  }
}

///|
test "credential framing rejects malformed zero oversized trailing and truncation" {
  for mode in 0..<5 {
    let peer = fixture_install(0)
    guard @credential.prepare_optional_inherited_inference_credential_v1()
      is Some(owner) else {
      abort("installed hostile credential was unavailable")
    }
    let secret : FixedArray[Byte] = [b'k']
    let frame = credential_frame(secret)
    if mode == 0 {
      frame[0] = b'X'
      assert_eq(fixture_send(peer, frame, 0, frame.length()), frame.length())
      ignore(fixture_close_write(peer))
    } else if mode == 1 {
      frame[8] = b'\x00'
      frame[9] = b'\x00'
      assert_eq(fixture_send(peer, frame, 0, 12), 12)
      ignore(fixture_close_write(peer))
    } else if mode == 2 {
      frame[8] = b'\x01'
      frame[9] = b'\x10'
      assert_eq(fixture_send(peer, frame, 0, 12), 12)
      ignore(fixture_close_write(peer))
    } else if mode == 3 {
      assert_eq(fixture_send(peer, frame, 0, frame.length()), frame.length())
      assert_eq(fixture_send(peer, secret, 0, 1), 1)
      ignore(fixture_close_write(peer))
    } else {
      assert_eq(fixture_send(peer, frame, 0, 5), 5)
      ignore(fixture_close_write(peer))
    }
    assert_eq(
      progress_until_terminal(owner),
      InheritedInferenceCredentialFailedProgress,
    )
    try
      ignore(owner.take_startup_auth_policy_v1(maximum_credential_bytes=4096))
    catch {
      InheritedInferenceCredentialFailed(CredentialFrameInvalid) => ()
      InheritedInferenceCredentialFailed(CredentialChannelLost) => ()
      _ => abort("hostile credential returned the wrong refusal")
    } noraise {
      _ => abort("hostile credential reached ready")
    }
    ignore(fixture_close(peer))
  }
}

///|
test "credential owner is singular and close is deterministic" {
  let peer = fixture_install(0)
  defer ignore(fixture_close(peer))
  guard @credential.prepare_optional_inherited_inference_credential_v1()
    is Some(owner) else {
    abort("installed credential was unavailable")
  }
  owner.close()
  owner.close()
  assert_eq(owner.progress(), InheritedInferenceCredentialFailedProgress)
}
