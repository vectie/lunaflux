///|
fn parent_attestation_fixture(
  canonical : Bytes,
) -> (FixedArray[Byte], Int) raise {
  let destination = FixedArray::make(
    parent_approval_attestation_capacity(),
    b'\x00',
  )
  let length = encode_parent_approval_attestation_v1(
    destination,
    manifest_digest_sha256=@crypto.bytes_to_hex_string(
      @crypto.sha256(canonical),
    ),
    key_identity_sha256="a".repeat(64),
    signature_digest_sha256="b".repeat(64),
    approved_source_sha256="c".repeat(64),
    bootstrap_digest_sha256="d".repeat(64),
    source_digest_sha256="e".repeat(64),
    model_generation=7UL,
    device_ordinal=2,
  )
  (destination, length)
}

///|
test "canonical absent parent approval remains standalone" {
  let encoded = FixedArray::make(
    parent_approval_attestation_capacity(),
    b'\xff',
  )
  let length = encode_absent_parent_approval_attestation_v1(encoded)
  assert_true(decode_parent_approval_attestation_v1(encoded, length) is None)
  encoded[9] = b'\x01'
  try decode_parent_approval_attestation_v1(encoded, length) catch {
    InvalidBootstrapSource(CanonicalEncoding) => ()
    _ => fail("noncanonical absent attestation returned the wrong refusal")
  } noraise {
    _ => fail("noncanonical absent attestation decoded")
  }
}

///|
test "parent approval attestation binds launch identity and consumes once" {
  let canonical = b"schema=lunaflux-parent-attestation-test.v1\n"
  let (encoded, length) = parent_attestation_fixture(canonical)
  guard decode_parent_approval_attestation_v1(encoded, length) is Some(witness) else {
    fail("present parent attestation decoded as absent")
  }
  witness.require_launch_identity(
    bootstrap_digest_sha256="d".repeat(64),
    source_digest_sha256="e".repeat(64),
    model_generation=7UL,
    device_ordinal=2,
  )
  ignore(
    witness.authenticate_manifest_claim(
      canonical,
      manifest_digest_sha256=@crypto.bytes_to_hex_string(
        @crypto.sha256(canonical),
      ),
      key_identity_sha256="a".repeat(64),
      signature_digest_sha256="b".repeat(64),
      approved_source_sha256="c".repeat(64),
    ),
  )
  try
    witness.authenticate_manifest_claim(
      canonical,
      manifest_digest_sha256=@crypto.bytes_to_hex_string(
        @crypto.sha256(canonical),
      ),
      key_identity_sha256="a".repeat(64),
      signature_digest_sha256="b".repeat(64),
      approved_source_sha256="c".repeat(64),
    )
  catch {
    Invalid(ExternalApproval) => ()
    _ => fail("replayed parent attestation returned the wrong refusal")
  } noraise {
    _ => fail("parent attestation authenticated twice")
  }
}

///|
test "parent approval attestation rejects stale generation and substitution" {
  let canonical = b"schema=lunaflux-parent-attestation-test.v1\n"
  let (encoded, length) = parent_attestation_fixture(canonical)
  guard decode_parent_approval_attestation_v1(encoded, length) is Some(witness) else {
    fail("present parent attestation decoded as absent")
  }
  try
    witness.require_launch_identity(
      bootstrap_digest_sha256="d".repeat(64),
      source_digest_sha256="e".repeat(64),
      model_generation=8UL,
      device_ordinal=2,
    )
  catch {
    Mismatch(ExternalApproval) => ()
    _ => fail("stale generation returned the wrong refusal")
  } noraise {
    _ => fail("stale generation entered child manifest admission")
  }
  encoded[16] = encoded[16].lxor(b'\x01')
  try decode_parent_approval_attestation_v1(encoded, length) catch {
    InvalidBootstrapSource(Integrity)
    | InvalidBootstrapSource(CanonicalEncoding) => ()
    _ => fail("substituted attestation returned the wrong refusal")
  } noraise {
    _ => fail("substituted attestation decoded")
  }
}
