///|
/// Independently versioned native inference protocol identity.
pub struct ProtocolVersion {
  priv wire_code : Int
} derive(Debug, Eq)

///|
/// Returns the only protocol version implemented by this package.
pub fn ProtocolVersion::v1() -> ProtocolVersion {
  { wire_code: 1 }
}

///|
/// Parses the stable integer wire representation, rejecting unknown versions.
pub fn ProtocolVersion::from_wire(
  wire_code : Int,
) -> ProtocolVersion raise InferenceContractError {
  if wire_code != 1 {
    raise invalid(ProtocolVersion, Unsupported, 0)
  }
  { wire_code, }
}

///|
pub fn ProtocolVersion::wire_code(self : ProtocolVersion) -> Int {
  self.wire_code
}

///|
/// Payload-safe stable identity of one externally submitted request.
pub struct RequestId {
  priv value : UInt64
} derive(Debug, Eq)

///|
pub fn RequestId::new(value : UInt64) -> RequestId raise InferenceContractError {
  let request_id : RequestId = { value, }
  request_id.validate()
  request_id
}

///|
pub fn RequestId::value(self : RequestId) -> UInt64 {
  self.value
}

///|
fn RequestId::validate(self : RequestId) -> Unit raise InferenceContractError {
  if self.value == 0UL {
    raise invalid(RequestId, OutOfRange, 0)
  }
}

///|
/// Monotonic cancellation generation shared by scheduler/worker contracts.
pub struct RequestGeneration {
  value : UInt64
} derive(Debug, Eq)

///|
pub fn RequestGeneration::initial() -> RequestGeneration {
  { value: 0UL }
}

///|
pub fn RequestGeneration::from_wire(value : UInt64) -> RequestGeneration {
  { value, }
}

///|
/// Returns the next generation, rejecting wraparound.
pub fn RequestGeneration::next(
  self : RequestGeneration,
) -> RequestGeneration raise InferenceContractError {
  if self.value == 0xffffffffffffffffUL {
    raise invalid(RequestGeneration, OutOfRange, 0)
  }
  { value: self.value + 1UL }
}

///|
pub fn RequestGeneration::value(self : RequestGeneration) -> UInt64 {
  self.value
}

///|
fn is_safe_identifier_byte(byte : Byte) -> Bool {
  (byte >= b'a' && byte <= b'z') ||
  (byte >= b'A' && byte <= b'Z') ||
  (byte >= b'0' && byte <= b'9') ||
  byte == b'-' ||
  byte == b'_' ||
  byte == b'.' ||
  byte == b':'
}

///|
fn checked_utf8_length(value : String) -> Int? {
  let units = value.code_units()
  let mut index = 0
  let mut byte_length = 0
  while index < units.length() {
    let unit = units[index].to_int()
    if unit >= 0xd800 && unit <= 0xdbff {
      if index + 1 >= units.length() {
        return None
      }
      let trailing = units[index + 1].to_int()
      if trailing < 0xdc00 || trailing > 0xdfff {
        return None
      }
      byte_length += 4
      index += 2
    } else if unit >= 0xdc00 && unit <= 0xdfff {
      return None
    } else {
      byte_length += if unit <= 0x7f {
        1
      } else if unit <= 0x7ff {
        2
      } else {
        3
      }
      index += 1
    }
  }
  Some(byte_length)
}

///|
fn bounded_identifier(
  value : BytesView,
  maximum : Int,
  field : ContractField,
) -> String raise InferenceContractError {
  if value.length() == 0 {
    raise invalid(field, Empty, 0)
  }
  if value.length() > maximum {
    raise invalid(field, TooLong, 0)
  }
  for byte in value {
    if !is_safe_identifier_byte(byte) {
      raise invalid(field, InvalidFormat, 0)
    }
  }
  @utf8.decode(value) catch {
    _ => raise invalid(field, InvalidFormat, 0)
  }
}

///|
fn validate_bounded_identifier(
  value : String,
  maximum : Int,
  field : ContractField,
) -> Unit raise InferenceContractError {
  if value.length() == 0 {
    raise invalid(field, Empty, 0)
  }
  if value.length() > maximum {
    raise invalid(field, TooLong, 0)
  }
  let byte_length = match checked_utf8_length(value) {
    Some(length) => length
    None => raise invalid(field, InvalidFormat, 0)
  }
  if byte_length > maximum {
    raise invalid(field, TooLong, 0)
  }
  for unit in value.code_units() {
    let code = unit.to_int()
    if !((code >= 0x61 && code <= 0x7a) ||
      (code >= 0x41 && code <= 0x5a) ||
      (code >= 0x30 && code <= 0x39) ||
      code == 0x2d ||
      code == 0x5f ||
      code == 0x2e ||
      code == 0x3a) {
      raise invalid(field, InvalidFormat, 0)
    }
  }
}

///|
/// Opaque cache security-scope identity. It deliberately has no `Debug`.
pub struct CacheScope {
  priv value : String
}

///|
pub fn CacheScope::from_ascii(
  value : BytesView,
  limits : InferenceLimits,
) -> CacheScope raise InferenceContractError {
  {
    value: bounded_identifier(value, limits.max_cache_scope_bytes(), CacheScope),
  }
}

///|
pub fn CacheScope::as_string(self : CacheScope) -> String {
  self.value
}

///|
fn CacheScope::validate(
  self : CacheScope,
  limits : InferenceLimits,
) -> Unit raise InferenceContractError {
  validate_bounded_identifier(
    self.value,
    limits.max_cache_scope_bytes(),
    CacheScope,
  )
}

///|
/// Opaque trace correlation. It deliberately has no `Debug`.
pub struct TraceCorrelation {
  priv value : String
}

///|
pub fn TraceCorrelation::from_ascii(
  value : BytesView,
  limits : InferenceLimits,
) -> TraceCorrelation raise InferenceContractError {
  {
    value: bounded_identifier(value, limits.max_trace_bytes(), TraceCorrelation),
  }
}

///|
pub fn TraceCorrelation::as_string(self : TraceCorrelation) -> String {
  self.value
}

///|
fn TraceCorrelation::validate(
  self : TraceCorrelation,
  limits : InferenceLimits,
) -> Unit raise InferenceContractError {
  validate_bounded_identifier(
    self.value,
    limits.max_trace_bytes(),
    TraceCorrelation,
  )
}
