///|
pub struct Fp8LaunchStorageIdentity(String) derive(Debug, Eq, Hash)

///|
fn is_lower_hex(value : StringView) -> Bool {
  if value.length() != 64 {
    return false
  }
  for code in value {
    if !(code >= '0' && code <= '9') && !(code >= 'a' && code <= 'f') {
      return false
    }
  }
  true
}

///|
pub fn Fp8LaunchStorageIdentity::from_sha256(
  value : String,
) -> Fp8LaunchStorageIdentity raise Fp8LaunchAbiError {
  if !is_lower_hex(value) {
    raise InvalidFp8LaunchAbi(InvalidStorageIdentity)
  }
  Fp8LaunchStorageIdentity(value)
}

///|
pub fn Fp8LaunchStorageIdentity::as_sha256(
  self : Fp8LaunchStorageIdentity,
) -> String {
  self.0
}

///|
pub struct Fp8LaunchStorageRegion {
  priv storage : Fp8LaunchStorageIdentity
  priv storage_bytes : Int64
  priv offset : Int64
  priv byte_length : Int64
  priv alignment : Int64
} derive(Debug, Eq)

///|
fn valid_alignment(value : Int64) -> Bool {
  value > 0L && value <= 1073741824L && (value & (value - 1L)) == 0L
}

///|
pub fn Fp8LaunchStorageRegion::new(
  storage~ : Fp8LaunchStorageIdentity,
  storage_bytes~ : Int64,
  offset~ : Int64,
  byte_length~ : Int64,
  alignment~ : Int64,
) -> Fp8LaunchStorageRegion raise Fp8LaunchAbiError {
  if storage_bytes <= 0L || offset < 0L || byte_length <= 0L {
    raise InvalidFp8LaunchAbi(InvalidStorageRegion)
  }
  if !valid_alignment(alignment) || offset % alignment != 0L {
    raise InvalidFp8LaunchAbi(RegionAlignment)
  }
  if offset > 9223372036854775807L - byte_length {
    raise InvalidFp8LaunchAbi(RegionOverflow)
  }
  if offset + byte_length > storage_bytes {
    raise InvalidFp8LaunchAbi(RegionOutOfBounds)
  }
  { storage, storage_bytes, offset, byte_length, alignment }
}

///|
pub fn Fp8LaunchStorageRegion::storage(
  self : Fp8LaunchStorageRegion,
) -> Fp8LaunchStorageIdentity {
  self.storage
}

///|
pub fn Fp8LaunchStorageRegion::storage_bytes(
  self : Fp8LaunchStorageRegion,
) -> Int64 {
  self.storage_bytes
}

///|
pub fn Fp8LaunchStorageRegion::offset(self : Fp8LaunchStorageRegion) -> Int64 {
  self.offset
}

///|
pub fn Fp8LaunchStorageRegion::byte_length(
  self : Fp8LaunchStorageRegion,
) -> Int64 {
  self.byte_length
}

///|
pub fn Fp8LaunchStorageRegion::alignment(
  self : Fp8LaunchStorageRegion,
) -> Int64 {
  self.alignment
}

///|
fn validate_distinct_regions(
  left : Fp8LaunchStorageRegion,
  right : Fp8LaunchStorageRegion,
) -> Unit raise Fp8LaunchAbiError {
  if left.storage != right.storage {
    return
  }
  if left.storage_bytes != right.storage_bytes {
    raise InvalidFp8LaunchAbi(RegionStorageMismatch)
  }
  let left_end = left.offset + left.byte_length
  let right_end = right.offset + right.byte_length
  if left.offset < right_end && right.offset < left_end {
    raise InvalidFp8LaunchAbi(RegionAlias)
  }
}
