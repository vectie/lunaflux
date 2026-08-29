///|
const LUNA_STARTUP_VERIFIER_KEY_HEADER_BYTES : Int = 16

///|
const LUNA_STARTUP_VERIFIER_KEY_MAX_PAYLOAD_BYTES : Int = 384

///|
fn wipe_startup_verifier_storage(
  header : FixedArray[Byte],
  payload : FixedArray[Byte],
) -> Unit {
  header.fill(b'\x00')
  payload.fill(b'\x00')
}

///|
fn startup_verifier_header_payload_length(
  header : FixedArray[Byte],
) -> (Int, Int, Int, Int)? {
  if header.length() != LUNA_STARTUP_VERIFIER_KEY_HEADER_BYTES ||
    header[0] != b'L' ||
    header[1] != b'F' ||
    header[2] != b'P' ||
    header[3] != b'1' ||
    header[4] != b'K' ||
    header[5] != b'E' ||
    header[6] != b'Y' ||
    header[7] != b'\n' ||
    header[11] != b'\x00' ||
    header[14] != b'\x00' ||
    header[15] != b'\x00' {
    return None
  }
  let key_length = header[8].to_int()
  let verifier_id_length = header[9].to_int()
  let authority_id_length = header[10].to_int()
  let payload_length = header[12].to_int() | (header[13].to_int() << 8)
  let expected = 64 + key_length + verifier_id_length + authority_id_length
  if key_length < 32 ||
    key_length > 64 ||
    verifier_id_length <= 0 ||
    verifier_id_length > 128 ||
    authority_id_length <= 0 ||
    authority_id_length > 128 ||
    payload_length != expected ||
    payload_length > LUNA_STARTUP_VERIFIER_KEY_MAX_PAYLOAD_BYTES {
    return None
  }
  Some((payload_length, key_length, verifier_id_length, authority_id_length))
}

///|
fn startup_verifier_copy_fixed(
  source : FixedArray[Byte],
  offset : Int,
  length : Int,
) -> FixedArray[Byte] {
  let output = FixedArray::make(length, b'\x00')
  for index in 0..<length {
    output[index] = source[offset + index]
  }
  output
}

///|
fn startup_verifier_decode_identifier(
  source : FixedArray[Byte],
  offset : Int,
  length : Int,
) -> String raise LunaCapabilityManifestError {
  let storage : Array[Byte] = []
  for index in 0..<length {
    storage.push(source[offset + index])
  }
  @utf8.decode(Bytes::from_array(storage)) catch {
    _ => raise Invalid(ExternalApproval)
  }
}

///|
pub fn prepare_optional_inherited_luna_startup_verifier_key_v1() -> LunaStartupVerifierKeyOwner? raise LunaCapabilityManifestError {
  let header = FixedArray::make(LUNA_STARTUP_VERIFIER_KEY_HEADER_BYTES, b'\x00')
  let payload = FixedArray::make(
    LUNA_STARTUP_VERIFIER_KEY_MAX_PAYLOAD_BYTES,
    b'\x00',
  )
  let native = @promotion_key.open_fixed()
  match @promotion_key.open_status(native) {
    0 =>
      Some({
        native,
        header,
        payload,
        header_cursor: 0,
        payload_cursor: 0,
        payload_length: 0,
        key_length: 0,
        verifier_id_length: 0,
        authority_id_length: 0,
        phase: 0,
      })
    1 => None
    _ => {
      wipe_startup_verifier_storage(header, payload)
      raise Invalid(ExternalApproval)
    }
  }
}

///|
fn LunaStartupVerifierKeyOwner::fail_closed(
  self : LunaStartupVerifierKeyOwner,
) -> LunaStartupVerifierKeyProgress {
  ignore(@promotion_key.close(self.native))
  wipe_startup_verifier_storage(self.header, self.payload)
  self.phase = 3
  LunaStartupVerifierKeyFailedProgress
}

///|
fn LunaStartupVerifierKeyOwner::read_header(
  self : LunaStartupVerifierKeyOwner,
) -> LunaStartupVerifierKeyProgress {
  let remaining = LUNA_STARTUP_VERIFIER_KEY_HEADER_BYTES - self.header_cursor
  let count = @promotion_key.try_read(
    self.native,
    self.header,
    self.header_cursor,
    remaining,
  )
  if count > 0 && count <= remaining {
    self.header_cursor += count
  } else if count == 0 {
    return LunaStartupVerifierKeyPending
  } else {
    return self.fail_closed()
  }
  if self.header_cursor != LUNA_STARTUP_VERIFIER_KEY_HEADER_BYTES {
    return LunaStartupVerifierKeyPending
  }
  guard startup_verifier_header_payload_length(self.header)
    is Some((payload, key, verifier, authority)) else {
    return self.fail_closed()
  }
  self.payload_length = payload
  self.key_length = key
  self.verifier_id_length = verifier
  self.authority_id_length = authority
  self.phase = 1
  LunaStartupVerifierKeyPending
}

///|
fn LunaStartupVerifierKeyOwner::read_payload(
  self : LunaStartupVerifierKeyOwner,
) -> LunaStartupVerifierKeyProgress {
  if self.payload_cursor < self.payload_length {
    let remaining = self.payload_length - self.payload_cursor
    let count = @promotion_key.try_read(
      self.native,
      self.payload,
      self.payload_cursor,
      remaining,
    )
    if count > 0 && count <= remaining {
      self.payload_cursor += count
    } else if count == 0 {
      return LunaStartupVerifierKeyPending
    } else {
      return self.fail_closed()
    }
  }
  if self.payload_cursor != self.payload_length {
    return LunaStartupVerifierKeyPending
  }
  match @promotion_key.peek(self.native) {
    -3 => {
      if @promotion_key.close(self.native) != 0 {
        return self.fail_closed()
      }
      self.phase = 2
      LunaStartupVerifierKeyReady
    }
    0 => LunaStartupVerifierKeyPending
    _ => self.fail_closed()
  }
}

///|
pub fn LunaStartupVerifierKeyOwner::progress(
  self : LunaStartupVerifierKeyOwner,
) -> LunaStartupVerifierKeyProgress {
  match self.phase {
    0 => self.read_header()
    1 => self.read_payload()
    2 => LunaStartupVerifierKeyReady
    _ => LunaStartupVerifierKeyFailedProgress
  }
}

///|
pub fn LunaStartupVerifierKeyOwner::take_handoff(
  self : LunaStartupVerifierKeyOwner,
) -> LunaStartupExternalApprovalHandoff raise LunaCapabilityManifestError {
  if self.phase != 2 || !@promotion_key.is_closed(self.native) {
    raise Invalid(ExternalApproval)
  }
  let rollout = startup_verifier_copy_fixed(self.payload, 0, 32)
  let tag = startup_verifier_copy_fixed(self.payload, 32, 32)
  let key = startup_verifier_copy_fixed(self.payload, 64, self.key_length)
  let verifier_offset = 64 + self.key_length
  let authority_offset = verifier_offset + self.verifier_id_length
  let verifier_key_result = Ok(
    startup_verifier_decode_identifier(
      self.payload,
      verifier_offset,
      self.verifier_id_length,
    ),
  ) catch {
    error => Err(error)
  }
  let verifier_key_id = match verifier_key_result {
    Ok(value) => value
    Err(error) => {
      key.fill(b'\x00')
      rollout.fill(b'\x00')
      tag.fill(b'\x00')
      wipe_startup_verifier_storage(self.header, self.payload)
      self.phase = 4
      raise error
    }
  }
  let authority_result = Ok(
    startup_verifier_decode_identifier(
      self.payload,
      authority_offset,
      self.authority_id_length,
    ),
  ) catch {
    error => Err(error)
  }
  let authority_id = match authority_result {
    Ok(value) => value
    Err(error) => {
      key.fill(b'\x00')
      rollout.fill(b'\x00')
      tag.fill(b'\x00')
      wipe_startup_verifier_storage(self.header, self.payload)
      self.phase = 4
      raise error
    }
  }
  let rollout_digest = LunaCapabilityManifestDigest::from_sha256(
    @crypto.bytes_to_hex_string(rollout),
  )
  let authentication_tag = LunaDetachedSignatureDigest::from_sha256(
    @crypto.bytes_to_hex_string(tag),
  )
  let policy_result = Ok(
    LunaExternalApprovalPolicy::new(
      schema=LUNA_EXTERNAL_APPROVAL_POLICY_SCHEMA,
      rollout_digest~,
      verifier_key_id~,
      authority_id~,
    ),
  ) catch {
    error => Err(error)
  }
  let policy = match policy_result {
    Ok(value) => value
    Err(error) => {
      key.fill(b'\x00')
      rollout.fill(b'\x00')
      tag.fill(b'\x00')
      wipe_startup_verifier_storage(self.header, self.payload)
      self.phase = 4
      raise error
    }
  }
  rollout.fill(b'\x00')
  tag.fill(b'\x00')
  wipe_startup_verifier_storage(self.header, self.payload)
  self.phase = 4
  {
    verifier: { policy, key, closed: false },
    rollout_digest,
    verifier_key_id,
    authority_id,
    authentication_tag,
    consumed: false,
  }
}

///|
pub fn LunaStartupVerifierKeyOwner::close(
  self : LunaStartupVerifierKeyOwner,
) -> Unit raise LunaCapabilityManifestError {
  if self.phase == 4 {
    return
  }
  wipe_startup_verifier_storage(self.header, self.payload)
  self.phase = 4
  if @promotion_key.close(self.native) != 0 {
    raise Invalid(ExternalApproval)
  }
}

///|
pub fn LunaStartupExternalApprovalHandoff::close(
  self : LunaStartupExternalApprovalHandoff,
) -> Unit {
  self.verifier.close()
  self.consumed = true
}

///|
fn LunaStartupExternalApprovalHandoff::authenticate_manifest_claim_once(
  self : LunaStartupExternalApprovalHandoff,
  canonical : Bytes,
  manifest_digest_sha256~ : String,
  key_identity_sha256~ : String,
  signature_digest_sha256~ : String,
  approved_source_sha256~ : String,
) -> LunaAuthenticatedExternalApproval raise LunaCapabilityManifestError {
  let manifest = LunaCapabilityManifestDigest::from_sha256(
    manifest_digest_sha256,
  )
  let key_identity = LunaSignatureKeyIdentityDigest::from_sha256(
    key_identity_sha256,
  )
  let signature = LunaDetachedSignatureDigest::from_sha256(
    signature_digest_sha256,
  )
  let source = LunaApprovedImmutableSourceIdentity::from_sha256(
    approved_source_sha256,
  )
  let envelope = LunaDetachedSignatureEnvelope::new(
    algorithm=LunaExternalEd25519V1,
    key_identity~,
    signature_digest=signature,
  )
  let signed = LunaExternalSignedApproval::new(
    schema=LUNA_EXTERNAL_SIGNED_APPROVAL_SCHEMA,
    rollout_digest=self.rollout_digest,
    verifier_key_id=self.verifier_key_id,
    authority_id=self.authority_id,
    manifest_digest=manifest,
    algorithm=LunaExternalEd25519V1,
    key_identity~,
    signature_digest=signature,
    approved_source=source,
    authentication_tag=self.authentication_tag,
  )
  self.verifier.verify(canonical, envelope, signed, source)
}

///|
/// Authenticates one exact inert launch/manifest claim, then wipes and closes
/// the startup key before returning opaque approval authority. Failure consumes
/// the handoff too, so stale aliases cannot replay it.
pub fn LunaStartupExternalApprovalHandoff::authenticate_manifest_claim(
  self : LunaStartupExternalApprovalHandoff,
  canonical : Bytes,
  manifest_digest_sha256~ : String,
  key_identity_sha256~ : String,
  signature_digest_sha256~ : String,
  approved_source_sha256~ : String,
) -> LunaAuthenticatedExternalApproval raise LunaCapabilityManifestError {
  if self.consumed {
    raise Invalid(ExternalApproval)
  }
  self.consumed = true
  let result : Result[
    LunaAuthenticatedExternalApproval,
    LunaCapabilityManifestError,
  ] = Ok(
    self.authenticate_manifest_claim_once(
      canonical,
      manifest_digest_sha256~,
      key_identity_sha256~,
      signature_digest_sha256~,
      approved_source_sha256~,
    ),
  ) catch {
    error => Err(error)
  }
  self.verifier.close()
  match result {
    Ok(value) => value
    Err(error) => raise error
  }
}
