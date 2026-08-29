///|
const PARENT_APPROVAL_ATTESTATION_BYTES : Int = 252

///|
pub struct WorkerParentApprovalSubjects {
  priv manifest_digest_sha256 : String
  priv key_identity_sha256 : String
  priv signature_digest_sha256 : String
  priv approved_source_sha256 : String
} derive(Debug, Eq)

///|
pub fn WorkerParentApprovalSubjects::from_luna_authorization(
  authorization : @luna_artifact_admission.LunaAotKernelAdmission,
) -> WorkerParentApprovalSubjects {
  {
    manifest_digest_sha256: authorization.manifest_digest().as_hex(),
    key_identity_sha256: authorization.approved_key_identity().as_hex(),
    signature_digest_sha256: authorization.detached_signature_digest().as_hex(),
    approved_source_sha256: authorization.approved_source_identity().as_hex(),
  }
}

///|
pub fn WorkerParentApprovalSubjects::from_fp8_authorization(
  authorization : @fp8_release_authority.Fp8ReusablePagedEnvelopeAuthorityV3,
) -> WorkerParentApprovalSubjects {
  {
    manifest_digest_sha256: authorization.manifest_digest().as_hex(),
    key_identity_sha256: authorization.approved_key_identity().as_hex(),
    signature_digest_sha256: authorization.detached_signature_digest().as_hex(),
    approved_source_sha256: authorization.approved_source_identity().as_hex(),
  }
}

///|
pub fn parent_approval_attestation_capacity() -> Int {
  PARENT_APPROVAL_ATTESTATION_BYTES
}

///|
fn parent_attestation_magic(storage : FixedArray[Byte]) -> Unit {
  storage[0] = b'L'
  storage[1] = b'F'
  storage[2] = b'P'
  storage[3] = b'A'
  storage[4] = b'1'
  storage[5] = b'\x00'
  storage[6] = b'\x00'
  storage[7] = b'\n'
}

///|
fn encode_parent_approval_attestation_v1(
  destination : FixedArray[Byte],
  manifest_digest_sha256~ : String,
  key_identity_sha256~ : String,
  signature_digest_sha256~ : String,
  approved_source_sha256~ : String,
  bootstrap_digest_sha256~ : String,
  source_digest_sha256~ : String,
  model_generation~ : UInt64,
  device_ordinal~ : Int,
) -> Int raise WorkerWireError {
  if destination.length() < PARENT_APPROVAL_ATTESTATION_BYTES ||
    model_generation == 0UL ||
    device_ordinal < 0 {
    raise InvalidBootstrapSource(CanonicalEncoding)
  }
  destination.fill(b'\x00')
  parent_attestation_magic(destination)
  destination[8] = b'\x01'
  write_bootstrap_source_digest(destination, 16, manifest_digest_sha256)
  write_bootstrap_source_digest(destination, 48, key_identity_sha256)
  write_bootstrap_source_digest(destination, 80, signature_digest_sha256)
  write_bootstrap_source_digest(destination, 112, approved_source_sha256)
  write_bootstrap_source_digest(destination, 144, bootstrap_digest_sha256)
  write_bootstrap_source_digest(destination, 176, source_digest_sha256)
  write_u64(destination, 208, model_generation)
  write_i32(destination, 216, device_ordinal)
  let checksum = bootstrap_source_digest_of(destination, 220)
  write_bootstrap_source_digest(destination, 220, checksum.as_hex())
  PARENT_APPROVAL_ATTESTATION_BYTES
}

///|
pub fn encode_parent_approval_subjects_attestation_v1(
  destination : FixedArray[Byte],
  subjects : WorkerParentApprovalSubjects,
  bootstrap_digest_sha256~ : String,
  source_digest_sha256~ : String,
  model_generation~ : UInt64,
  device_ordinal~ : Int,
) -> Int raise WorkerWireError {
  encode_parent_approval_attestation_v1(
    destination,
    manifest_digest_sha256=subjects.manifest_digest_sha256,
    key_identity_sha256=subjects.key_identity_sha256,
    signature_digest_sha256=subjects.signature_digest_sha256,
    approved_source_sha256=subjects.approved_source_sha256,
    bootstrap_digest_sha256~,
    source_digest_sha256~,
    model_generation~,
    device_ordinal~,
  )
}

///|
pub fn encode_absent_parent_approval_attestation_v1(
  destination : FixedArray[Byte],
) -> Int raise WorkerWireError {
  if destination.length() < PARENT_APPROVAL_ATTESTATION_BYTES {
    raise InvalidBootstrapSource(CanonicalEncoding)
  }
  destination.fill(b'\x00')
  parent_attestation_magic(destination)
  let checksum = bootstrap_source_digest_of(destination, 220)
  write_bootstrap_source_digest(destination, 220, checksum.as_hex())
  PARENT_APPROVAL_ATTESTATION_BYTES
}

///|
pub fn decode_parent_approval_attestation_v1(
  source : FixedArray[Byte],
  length : Int,
) -> @luna_capability_manifest.LunaParentStartupApprovalWitness? raise WorkerWireError {
  if length != PARENT_APPROVAL_ATTESTATION_BYTES ||
    source.length() < length ||
    source[0] != b'L' ||
    source[1] != b'F' ||
    source[2] != b'P' ||
    source[3] != b'A' ||
    source[4] != b'1' ||
    source[5] != b'\x00' ||
    source[6] != b'\x00' ||
    source[7] != b'\n' ||
    bootstrap_source_raw_digest(source, 220) !=
    bootstrap_source_digest_of(source, 220) {
    raise InvalidBootstrapSource(CanonicalEncoding)
  }
  if source[8] == b'\x00' {
    for index in 9..<220 {
      if source[index] != b'\x00' {
        raise InvalidBootstrapSource(CanonicalEncoding)
      }
    }
    return None
  }
  if source[8] != b'\x01' {
    raise InvalidBootstrapSource(CanonicalEncoding)
  }
  for index in 9..<16 {
    if source[index] != b'\x00' {
      raise InvalidBootstrapSource(CanonicalEncoding)
    }
  }
  Some(
    @luna_capability_manifest.LunaParentStartupApprovalWitness::from_private_parent_channel_v1(
      manifest_digest_sha256=bootstrap_source_raw_digest(source, 16).as_hex(),
      key_identity_sha256=bootstrap_source_raw_digest(source, 48).as_hex(),
      signature_digest_sha256=bootstrap_source_raw_digest(source, 80).as_hex(),
      approved_source_sha256=bootstrap_source_raw_digest(source, 112).as_hex(),
      bootstrap_digest_sha256=bootstrap_source_raw_digest(source, 144).as_hex(),
      source_digest_sha256=bootstrap_source_raw_digest(source, 176).as_hex(),
      model_generation=read_u64(source, 208),
      device_ordinal=read_i32(source, 216),
    ) catch {
      _ => raise InvalidBootstrapSource(CanonicalEncoding)
    },
  )
}
