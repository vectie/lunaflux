///|
test "external record admission binds exact bytes envelope and approval" {
  let canonical = b"schema=lunaflux-test-external-record.v1\nvalue=exact\n"
  let digest = @crypto.bytes_to_hex_string(@crypto.sha256(canonical))
  let manifest_digest = LunaCapabilityManifestDigest::from_sha256(digest)
  let key = LunaSignatureKeyIdentityDigest::from_sha256("a".repeat(64))
  let signature = LunaDetachedSignatureDigest::from_sha256("b".repeat(64))
  let source = LunaApprovedImmutableSourceIdentity::from_sha256("c".repeat(64))
  let envelope = LunaDetachedSignatureEnvelope::new(
    algorithm=LunaExternalEd25519V1,
    key_identity=key,
    signature_digest=signature,
  )
  let receipt = luna_test_authenticated_external_approval(
    canonical, envelope, source,
  )
  let admitted = admit_luna_external_approved_record(
    canonical,
    envelope,
    receipt,
    maximum_canonical_bytes=4096,
  )
  assert_eq(admitted.record_digest(), manifest_digest)
  assert_eq(admitted.approved_source_identity(), source)
  assert_eq(admitted.approved_key_identity(), key)
  assert_eq(admitted.detached_signature_digest(), signature)
}

///|
test "external record admission rejects mutation envelope drift and limits" {
  let canonical = b"schema=lunaflux-test-external-record.v1\nvalue=exact\n"
  let key = LunaSignatureKeyIdentityDigest::from_sha256("a".repeat(64))
  let signature = LunaDetachedSignatureDigest::from_sha256("b".repeat(64))
  let source = LunaApprovedImmutableSourceIdentity::from_sha256("c".repeat(64))
  let exact = LunaDetachedSignatureEnvelope::new(
    algorithm=LunaExternalEd25519V1,
    key_identity=key,
    signature_digest=signature,
  )
  let receipt = luna_test_authenticated_external_approval(
    canonical, exact, source,
  )
  for mutated in [b"schema=lunaflux-test-external-record.v1\nvalue=changed\n"] {
    try
      admit_luna_external_approved_record(
        mutated,
        exact,
        receipt,
        maximum_canonical_bytes=4096,
      )
    catch {
      Mismatch(Digest) => ()
      _ => fail("unexpected mutated-record rejection")
    } noraise {
      _ => fail("mutated approved record was admitted")
    }
  }
  let drifted = LunaDetachedSignatureEnvelope::new(
    algorithm=LunaExternalEd25519V1,
    key_identity=key,
    signature_digest=LunaDetachedSignatureDigest::from_sha256("d".repeat(64)),
  )
  try
    admit_luna_external_approved_record(
      canonical,
      drifted,
      receipt,
      maximum_canonical_bytes=4096,
    )
  catch {
    Mismatch(ExternalApproval) => ()
    _ => fail("unexpected envelope-drift rejection")
  } noraise {
    _ => fail("drifted signature envelope was admitted")
  }
  try
    admit_luna_external_approved_record(
      canonical,
      exact,
      receipt,
      maximum_canonical_bytes=1,
    )
  catch {
    Invalid(CanonicalBytes) => ()
    _ => fail("unexpected external-record limit rejection")
  } noraise {
    _ => fail("oversized external record was admitted")
  }
}
