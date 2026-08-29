///|
/// White-box-only fixture: production code has no arbitrary-key constructor.
fn luna_test_authenticated_external_approval(
  canonical : Bytes,
  envelope : LunaDetachedSignatureEnvelope,
  approved_source : LunaApprovedImmutableSourceIdentity,
) -> LunaAuthenticatedExternalApproval raise LunaCapabilityManifestError {
  let rollout_digest = LunaCapabilityManifestDigest::from_sha256("d".repeat(64))
  let policy = LunaExternalApprovalPolicy::new(
    schema=LUNA_EXTERNAL_APPROVAL_POLICY_SCHEMA,
    rollout_digest~,
    verifier_key_id="test-key",
    authority_id="test-authority",
  )
  let key = b"0123456789abcdef0123456789abcdef".to_fixedarray()
  let manifest_digest = LunaCapabilityManifestDigest(
    @crypto.bytes_to_hex_string(@crypto.sha256(canonical)),
  )
  let unsigned : LunaExternalSignedApproval = {
    schema: LUNA_EXTERNAL_SIGNED_APPROVAL_SCHEMA,
    rollout_digest,
    verifier_key_id: "test-key",
    authority_id: "test-authority",
    manifest_digest,
    algorithm: envelope.algorithm,
    key_identity: envelope.key_identity,
    signature_digest: envelope.signature_digest,
    approved_source,
    authentication_tag: LunaDetachedSignatureDigest::from_sha256("0".repeat(64)),
  }
  let digest = luna_external_approval_hmac(key, unsigned.signing_bytes())
  let signed : LunaExternalSignedApproval = {
    ..unsigned,
    authentication_tag: LunaDetachedSignatureDigest::from_sha256(
      @crypto.bytes_to_hex_string(digest),
    ),
  }
  digest.fill(b'\x00')
  let verifier : LunaExternalApprovalVerifier = { policy, key, closed: false }
  let approval = verifier.verify(canonical, envelope, signed, approved_source)
  verifier.close()
  approval
}

///|
fn luna_test_startup_handoff(
  canonical : Bytes,
  key_identity : LunaSignatureKeyIdentityDigest,
  signature : LunaDetachedSignatureDigest,
  source : LunaApprovedImmutableSourceIdentity,
) -> LunaStartupExternalApprovalHandoff raise LunaCapabilityManifestError {
  let rollout_digest = LunaCapabilityManifestDigest::from_sha256("d".repeat(64))
  let policy = LunaExternalApprovalPolicy::new(
    schema=LUNA_EXTERNAL_APPROVAL_POLICY_SCHEMA,
    rollout_digest~,
    verifier_key_id="test-key",
    authority_id="test-authority",
  )
  let key = b"0123456789abcdef0123456789abcdef".to_fixedarray()
  let manifest_digest = LunaCapabilityManifestDigest(
    @crypto.bytes_to_hex_string(@crypto.sha256(canonical)),
  )
  let unsigned : LunaExternalSignedApproval = {
    schema: LUNA_EXTERNAL_SIGNED_APPROVAL_SCHEMA,
    rollout_digest,
    verifier_key_id: "test-key",
    authority_id: "test-authority",
    manifest_digest,
    algorithm: LunaExternalEd25519V1,
    key_identity,
    signature_digest: signature,
    approved_source: source,
    authentication_tag: LunaDetachedSignatureDigest::from_sha256("0".repeat(64)),
  }
  let tag = luna_external_approval_hmac(key, unsigned.signing_bytes())
  let authentication_tag = LunaDetachedSignatureDigest::from_sha256(
    @crypto.bytes_to_hex_string(tag),
  )
  tag.fill(b'\x00')
  {
    verifier: { policy, key, closed: false },
    rollout_digest,
    verifier_key_id: "test-key",
    authority_id: "test-authority",
    authentication_tag,
    consumed: false,
  }
}

///|
test "startup verifier handoff is one-shot and wipes every key alias" {
  let canonical = b"schema=lunaflux-test-startup-handoff.v1\nvalue=exact\n"
  let identity = LunaSignatureKeyIdentityDigest::from_sha256("a".repeat(64))
  let signature = LunaDetachedSignatureDigest::from_sha256("b".repeat(64))
  let source = LunaApprovedImmutableSourceIdentity::from_sha256("c".repeat(64))
  let handoff = luna_test_startup_handoff(
    canonical, identity, signature, source,
  )
  ignore(
    handoff.authenticate_manifest_claim(
      canonical,
      manifest_digest_sha256=@crypto.bytes_to_hex_string(
        @crypto.sha256(canonical),
      ),
      key_identity_sha256=identity.as_hex(),
      signature_digest_sha256=signature.as_hex(),
      approved_source_sha256=source.as_hex(),
    ),
  )
  assert_true(handoff.consumed)
  assert_true(handoff.verifier.closed)
  for byte in handoff.verifier.key {
    assert_eq(byte, b'\x00')
  }
  try
    handoff.authenticate_manifest_claim(
      canonical,
      manifest_digest_sha256=@crypto.bytes_to_hex_string(
        @crypto.sha256(canonical),
      ),
      key_identity_sha256=identity.as_hex(),
      signature_digest_sha256=signature.as_hex(),
      approved_source_sha256=source.as_hex(),
    )
  catch {
    Invalid(ExternalApproval) => ()
    _ => fail("replayed startup handoff returned the wrong refusal")
  } noraise {
    _ => fail("startup verifier handoff authenticated twice")
  }
}

///|
test "startup verifier handoff fails closed on a substituted manifest" {
  let canonical = b"schema=lunaflux-test-startup-handoff.v1\nvalue=exact\n"
  let identity = LunaSignatureKeyIdentityDigest::from_sha256("a".repeat(64))
  let signature = LunaDetachedSignatureDigest::from_sha256("b".repeat(64))
  let source = LunaApprovedImmutableSourceIdentity::from_sha256("c".repeat(64))
  let handoff = luna_test_startup_handoff(
    canonical, identity, signature, source,
  )
  try
    handoff.authenticate_manifest_claim(
      b"schema=lunaflux-test-startup-handoff.v1\nvalue=substituted\n",
      manifest_digest_sha256=@crypto.bytes_to_hex_string(
        @crypto.sha256(canonical),
      ),
      key_identity_sha256=identity.as_hex(),
      signature_digest_sha256=signature.as_hex(),
      approved_source_sha256=source.as_hex(),
    )
  catch {
    Mismatch(ExternalApproval) => ()
    _ => fail("substituted manifest returned the wrong refusal")
  } noraise {
    _ => fail("substituted manifest gained approval authority")
  }
  assert_true(handoff.consumed)
  assert_true(handoff.verifier.closed)
  for byte in handoff.verifier.key {
    assert_eq(byte, b'\x00')
  }
}

///|
test "startup verifier handoff wipes key when hostile subjects fail parsing" {
  let canonical = b"schema=lunaflux-test-startup-handoff.v1\nvalue=exact\n"
  let identity = LunaSignatureKeyIdentityDigest::from_sha256("a".repeat(64))
  let signature = LunaDetachedSignatureDigest::from_sha256("b".repeat(64))
  let source = LunaApprovedImmutableSourceIdentity::from_sha256("c".repeat(64))
  let handoff = luna_test_startup_handoff(
    canonical, identity, signature, source,
  )
  try
    handoff.authenticate_manifest_claim(
      canonical,
      manifest_digest_sha256="not-a-digest",
      key_identity_sha256=identity.as_hex(),
      signature_digest_sha256=signature.as_hex(),
      approved_source_sha256=source.as_hex(),
    )
  catch {
    Invalid(Digest) => ()
    _ => fail("hostile subject returned the wrong refusal")
  } noraise {
    _ => fail("hostile subject gained approval authority")
  }
  assert_true(handoff.consumed)
  assert_true(handoff.verifier.closed)
  for byte in handoff.verifier.key {
    assert_eq(byte, b'\x00')
  }
}

///|
test "authenticated approval rejects hostile tag and closed verifier and wipes key" {
  let canonical = b"schema=lunaflux-test-authenticated-approval.v1\nvalue=exact\n"
  let rollout_digest = LunaCapabilityManifestDigest::from_sha256("d".repeat(64))
  let policy = LunaExternalApprovalPolicy::new(
    schema=LUNA_EXTERNAL_APPROVAL_POLICY_SCHEMA,
    rollout_digest~,
    verifier_key_id="test-key",
    authority_id="test-authority",
  )
  let key = b"0123456789abcdef0123456789abcdef".to_fixedarray()
  let identity = LunaSignatureKeyIdentityDigest::from_sha256("a".repeat(64))
  let signature = LunaDetachedSignatureDigest::from_sha256("b".repeat(64))
  let source = LunaApprovedImmutableSourceIdentity::from_sha256("c".repeat(64))
  let envelope = LunaDetachedSignatureEnvelope::new(
    algorithm=LunaExternalEd25519V1,
    key_identity=identity,
    signature_digest=signature,
  )
  let hostile = LunaExternalSignedApproval::new(
    schema=LUNA_EXTERNAL_SIGNED_APPROVAL_SCHEMA,
    rollout_digest~,
    verifier_key_id="test-key",
    authority_id="test-authority",
    manifest_digest=LunaCapabilityManifestDigest(
      @crypto.bytes_to_hex_string(@crypto.sha256(canonical)),
    ),
    algorithm=LunaExternalEd25519V1,
    key_identity=identity,
    signature_digest=signature,
    approved_source=source,
    authentication_tag=LunaDetachedSignatureDigest::from_sha256("0".repeat(64)),
  )
  let verifier : LunaExternalApprovalVerifier = { policy, key, closed: false }
  try verifier.verify(canonical, envelope, hostile, source) catch {
    Mismatch(ExternalApproval) => ()
    _ => fail("unexpected hostile authentication-tag rejection")
  } noraise {
    _ => fail("unauthenticated external evidence minted approval authority")
  }
  verifier.close()
  assert_true(verifier.closed)
  for byte in verifier.key {
    assert_eq(byte, b'\x00')
  }
  try verifier.verify(canonical, envelope, hostile, source) catch {
    Invalid(ExternalApproval) => ()
    _ => fail("unexpected closed-verifier rejection")
  } noraise {
    _ => fail("closed verifier admitted external evidence")
  }
}

///|
test "authenticated approval binds immutable policy and source subjects" {
  let canonical = b"schema=lunaflux-test-authenticated-approval.v1\nvalue=exact\n"
  let identity = LunaSignatureKeyIdentityDigest::from_sha256("a".repeat(64))
  let signature = LunaDetachedSignatureDigest::from_sha256("b".repeat(64))
  let source = LunaApprovedImmutableSourceIdentity::from_sha256("c".repeat(64))
  let envelope = LunaDetachedSignatureEnvelope::new(
    algorithm=LunaExternalEd25519V1,
    key_identity=identity,
    signature_digest=signature,
  )
  let approval = luna_test_authenticated_external_approval(
    canonical, envelope, source,
  )
  let admitted = admit_luna_external_approved_record(
    canonical,
    envelope,
    approval,
    maximum_canonical_bytes=4096,
  )
  assert_eq(admitted.approved_source_identity(), source)
  try
    admit_luna_external_approved_record(
      canonical,
      LunaDetachedSignatureEnvelope::new(
        algorithm=LunaExternalEd25519V1,
        key_identity=identity,
        signature_digest=LunaDetachedSignatureDigest::from_sha256(
          "e".repeat(64),
        ),
      ),
      approval,
      maximum_canonical_bytes=4096,
    )
  catch {
    Mismatch(ExternalApproval) => ()
    _ => fail("unexpected exact-envelope binding rejection")
  } noraise {
    _ => fail("authenticated approval replayed across envelope subjects")
  }
}
