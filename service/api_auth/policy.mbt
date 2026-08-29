///|
fn luna_api_auth_error(
  rule : LunaApiAuthRule,
  issue : LunaApiAuthIssue,
) -> LunaApiAuthError {
  LunaApiAuthFailed(rule~, issue~)
}

///|
fn luna_api_auth_range_is_valid(
  source_length : Int,
  source_offset : Int,
  length : Int,
) -> Bool {
  source_offset >= 0 &&
  length >= 0 &&
  source_offset <= source_length &&
  length <= source_length - source_offset
}

///|
pub fn LunaApiAuthPolicy::required_byte_cells(
  maximum_credential_bytes : Int,
) -> UInt64 raise LunaApiAuthError {
  if maximum_credential_bytes <= 0 {
    raise luna_api_auth_error(LunaApiAuthPolicy, LunaApiAuthInvalid)
  }
  if maximum_credential_bytes > LUNA_API_AUTH_MAX_CREDENTIAL_BYTES {
    raise luna_api_auth_error(LunaApiAuthPolicy, LunaApiAuthTooLong)
  }
  maximum_credential_bytes.to_uint64()
}

///|
/// Copies one nonempty expected credential into startup-owned fixed storage.
///
/// Validation authenticates the configured policy bound before the source
/// range and credential length. The hard supported bound is 4096 bytes.
pub fn LunaApiAuthPolicy::new(
  expected : FixedArray[Byte],
  source_offset~ : Int,
  length~ : Int,
  maximum_credential_bytes~ : Int,
) -> LunaApiAuthPolicy raise LunaApiAuthError {
  ignore(LunaApiAuthPolicy::required_byte_cells(maximum_credential_bytes))
  if !luna_api_auth_range_is_valid(expected.length(), source_offset, length) {
    raise luna_api_auth_error(LunaApiAuthSource, LunaApiAuthOutOfRange)
  }
  if length == 0 {
    raise luna_api_auth_error(LunaApiAuthCredential, LunaApiAuthEmpty)
  }
  if length > maximum_credential_bytes {
    raise luna_api_auth_error(LunaApiAuthCredential, LunaApiAuthTooLong)
  }
  let owned : FixedArray[Byte] = FixedArray::make(length, 0)
  for index in 0..<length {
    owned[index] = expected[source_offset + index]
  }
  {
    expected: owned,
    expected_length: length,
    maximum_credential_bytes,
    closed: false,
  }
}

///|
/// Idempotently destroys the retained credential. All policy aliases share
/// the owned fixed storage, so closing the singular owner invalidates stale
/// verifier aliases without exposing credential or storage state.
pub fn LunaApiAuthPolicy::close(self : LunaApiAuthPolicy) -> Unit {
  if self.closed {
    return
  }
  for index in 0..<self.expected.length() {
    self.expected[index] = b'\x00'
  }
  self.expected_length = 0
  self.maximum_credential_bytes = 0
  self.closed = true
}

///|
fn LunaApiAuthPolicy::is_authenticated(self : LunaApiAuthPolicy) -> Bool {
  !self.closed &&
  self.maximum_credential_bytes > 0 &&
  self.maximum_credential_bytes <= LUNA_API_AUTH_MAX_CREDENTIAL_BYTES &&
  self.expected.length() == self.expected_length &&
  self.expected_length > 0 &&
  self.expected_length <= self.maximum_credential_bytes
}

///|
fn LunaApiAuthPolicy::compare_byte(
  self : LunaApiAuthPolicy,
  index : Int,
  actual : Byte,
  difference : UInt,
) -> UInt {
  difference | (self.expected[index].to_uint() ^ actual.to_uint())
}

///|
fn LunaApiAuthPolicy::luna_api_auth_compare(
  self : LunaApiAuthPolicy,
  source : FixedArray[Byte],
  source_offset : Int,
  length : Int,
) -> LunaApiAuthComparison {
  let mut difference : UInt = if length == self.expected_length { 0 } else { 1 }
  let mut comparison_count = 0
  for index in 0..<self.expected_length {
    let actual : Byte = if index < length {
      source[source_offset + index]
    } else {
      0
    }
    difference = self.compare_byte(index, actual, difference)
    comparison_count += 1
  }
  { difference, comparison_count }
}

///|
/// Authenticates a caller-owned credential range without retaining it.
///
/// The policy invariant is checked first, followed by the subtraction-safe
/// source range. Every valid range then executes exactly
/// `expected_length` scalar byte comparisons. The configured maximum remains
/// an ingress bound, not padding work. A length mismatch is folded into the
/// accumulated difference; excess bytes are ignored, missing bytes compare as
/// zero, and no mismatch ends the credential-length loop early.
pub fn LunaApiAuthPolicy::verify(
  self : LunaApiAuthPolicy,
  source : FixedArray[Byte],
  source_offset~ : Int,
  length~ : Int,
) -> LunaApiAuthDecision raise LunaApiAuthError {
  if !self.is_authenticated() {
    raise luna_api_auth_error(LunaApiAuthPolicy, LunaApiAuthInvalid)
  }
  if !luna_api_auth_range_is_valid(source.length(), source_offset, length) {
    raise luna_api_auth_error(LunaApiAuthSource, LunaApiAuthOutOfRange)
  }
  let comparison = self.luna_api_auth_compare(source, source_offset, length)
  {
    state: if comparison.difference == 0 &&
      comparison.comparison_count == self.expected_length {
      0
    } else {
      1
    },
  }
}
