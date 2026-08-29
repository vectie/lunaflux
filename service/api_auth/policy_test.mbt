///|
fn expect_api_auth_error(
  action : () -> Unit raise LunaApiAuthError,
  rule : LunaApiAuthRule,
  issue : LunaApiAuthIssue,
) -> Unit raise {
  try action() catch {
    LunaApiAuthFailed(rule=observed_rule, issue=observed_issue) =>
      if observed_rule != rule || observed_issue != issue {
        fail("unexpected API authentication error")
      }
  } noraise {
    _ => fail("expected API authentication error")
  }
}

///|
test "Luna API authentication copies and verifies bounded ranges" {
  let expected : FixedArray[Byte] = [b'x', b'a', b'b', b'c', b'y']
  let policy = LunaApiAuthPolicy::new(
    expected,
    source_offset=1,
    length=3,
    maximum_credential_bytes=8,
  )
  let retained_source_alias = expected
  retained_source_alias[1] = b'z'
  let exact : FixedArray[Byte] = [b'q', b'a', b'b', b'c', b'r']
  assert_true(policy.verify(exact, source_offset=1, length=3).is_accepted())
  assert_true(policy.verify(expected, source_offset=1, length=3).is_rejected())
  assert_true(policy.verify(exact, source_offset=1, length=2).is_rejected())
  assert_true(policy.verify(exact, source_offset=0, length=3).is_rejected())
  assert_true(policy.verify(exact, source_offset=2, length=3).is_rejected())
}

///|
test "Luna API authentication enforces startup bounds and precedence" {
  let empty : FixedArray[Byte] = []
  expect_api_auth_error(
    () => {
      ignore(
        LunaApiAuthPolicy::new(
          empty,
          source_offset=0,
          length=0,
          maximum_credential_bytes=8,
        ),
      )
    },
    LunaApiAuthCredential,
    LunaApiAuthEmpty,
  )
  let maximum = FixedArray::make(4096, b'a')
  let maximum_policy = LunaApiAuthPolicy::new(
    maximum,
    source_offset=0,
    length=4096,
    maximum_credential_bytes=4096,
  )
  assert_true(
    maximum_policy.verify(maximum, source_offset=0, length=4096).is_accepted(),
  )
  let too_long = FixedArray::make(4097, b'a')
  expect_api_auth_error(
    () => {
      ignore(
        LunaApiAuthPolicy::new(
          too_long,
          source_offset=0,
          length=4097,
          maximum_credential_bytes=4096,
        ),
      )
    },
    LunaApiAuthCredential,
    LunaApiAuthTooLong,
  )
  expect_api_auth_error(
    () => {
      ignore(
        LunaApiAuthPolicy::new(
          too_long,
          source_offset=5000,
          length=4097,
          maximum_credential_bytes=4096,
        ),
      )
    },
    LunaApiAuthSource,
    LunaApiAuthOutOfRange,
  )
  expect_api_auth_error(
    () => ignore(LunaApiAuthPolicy::required_byte_cells(0)),
    LunaApiAuthPolicy,
    LunaApiAuthInvalid,
  )
  expect_api_auth_error(
    () => ignore(LunaApiAuthPolicy::required_byte_cells(4097)),
    LunaApiAuthPolicy,
    LunaApiAuthTooLong,
  )
}

///|
test "Luna API authentication rejects invalid caller ranges before length" {
  let expected : FixedArray[Byte] = [b'a', b'b', b'c']
  let policy = LunaApiAuthPolicy::new(
    expected,
    source_offset=0,
    length=3,
    maximum_credential_bytes=8,
  )
  for
    item in [(-1, 3), (0, -1), (4, 0), (2, 2), (2147483647, 1), (0, 2147483647)] {
    let (offset, length) = item
    expect_api_auth_error(
      () => ignore(policy.verify(expected, source_offset=offset, length~)),
      LunaApiAuthSource,
      LunaApiAuthOutOfRange,
    )
  }
}
