///|
/// Independently versioned native inference protocol identity.
pub struct ProtocolVersion {
  wire_code : Int
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
  value : UInt64
} derive(Debug, Eq)

///|
pub fn RequestId::new(value : UInt64) -> RequestId raise InferenceContractError {
  if value == 0UL {
    raise invalid(RequestId, OutOfRange, 0)
  }
  { value, }
}

///|
pub fn RequestId::value(self : RequestId) -> UInt64 {
  self.value
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
/// Opaque cache security-scope identity. It deliberately has no `Debug`.
pub struct CacheScope {
  value : String
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
/// Opaque trace correlation. It deliberately has no `Debug`.
pub struct TraceCorrelation {
  value : String
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
