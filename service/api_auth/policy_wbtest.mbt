///|
test "Luna API authentication compares the fixed credential length" {
  let expected : FixedArray[Byte] = [b'a', b'b', b'c', b'd']
  let policy = LunaApiAuthPolicy::new(
    expected,
    source_offset=0,
    length=4,
    maximum_credential_bytes=8,
  )
  for mismatch in 0..<4 {
    let source : FixedArray[Byte] = [b'a', b'b', b'c', b'd']
    source[mismatch] = source[mismatch] + 1
    let comparison = policy.luna_api_auth_compare(source, 0, 4)
    assert_true(comparison.difference != 0)
    assert_eq(comparison.comparison_count, 4)
  }
  let exact = policy.luna_api_auth_compare(expected, 0, 4)
  assert_eq(exact.difference, 0U)
  assert_eq(exact.comparison_count, 4)
  let varied : FixedArray[Byte] = [
    b'a', b'b', b'c', b'd', b'e', b'f', b'g', b'h', b'i',
  ]
  for length in [0, 2, 5, 8, 9] {
    let comparison = policy.luna_api_auth_compare(varied, 0, length)
    assert_true(comparison.difference != 0)
    assert_eq(comparison.comparison_count, 4)
    assert_true(policy.verify(varied, source_offset=0, length~).is_rejected())
  }
  let varied_tail : FixedArray[Byte] = [
    b'a', b'b', b'c', b'd', b'e', b'f', b'g', b'h', b'z',
  ]
  assert_eq(
    policy.luna_api_auth_compare(varied, 0, 9).difference,
    policy.luna_api_auth_compare(varied_tail, 0, 9).difference,
  )
}

///|
test "Luna API authentication policy invariant precedes source validation" {
  let expected : FixedArray[Byte] = [b'a']
  let forged : LunaApiAuthPolicy = {
    expected,
    expected_length: 0,
    maximum_credential_bytes: 1,
    closed: false,
  }
  try forged.verify(expected, source_offset=-1, length=2) catch {
    LunaApiAuthFailed(rule=LunaApiAuthPolicy, issue=LunaApiAuthInvalid) => ()
    _ => fail("unexpected forged policy failure")
  } noraise {
    _ => fail("forged policy was accepted")
  }
}

///|
test "Luna API authentication public verifier uses the fixed comparison" {
  let expected : FixedArray[Byte] = [b'a', b'b', b'c', b'd']
  let policy = LunaApiAuthPolicy::new(
    expected,
    source_offset=0,
    length=4,
    maximum_credential_bytes=8,
  )
  for _ in 0..<64 {
    assert_true(
      policy.verify(expected, source_offset=0, length=4).is_accepted(),
    )
  }
}

///|
test "Luna API authentication close wipes every retained cell and aliases" {
  let policy = LunaApiAuthPolicy::new(
    [b's', b'e', b'c', b'r', b'e', b't'],
    source_offset=0,
    length=6,
    maximum_credential_bytes=8,
  )
  let stale = policy
  policy.close()
  policy.close()
  for byte in policy.expected {
    assert_eq(byte, b'\x00')
  }
  assert_eq(policy.expected_length, 0)
  assert_eq(policy.maximum_credential_bytes, 0)
  assert_true(policy.closed)
  try stale.verify([b's'], source_offset=0, length=1) catch {
    LunaApiAuthFailed(rule=LunaApiAuthPolicy, issue=LunaApiAuthInvalid) => ()
    _ => fail("unexpected closed policy rejection")
  } noraise {
    _ => fail("stale policy alias survived close")
  }
}
