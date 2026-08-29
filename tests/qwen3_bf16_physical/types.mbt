///|
suberror Qwen3PhysicalError {
  InvalidInput(String)
  IdentityMismatch(String)
  CorpusMismatch(String)
  CandidateMismatch(String)
  NumericMaterialization(String)
  DeviceMismatch(String)
} derive(Debug)

///|
fn qwen3_require(
  condition : Bool,
  message : String,
) -> Unit raise Qwen3PhysicalError {
  if !condition {
    raise InvalidInput(message)
  }
}

///|
fn qwen3_is_sha256(value : String) -> Bool {
  value.length() == 64 &&
  value.all(character => {
    character.is_ascii_digit() || (character >= 'a' && character <= 'f')
  })
}

///|
fn qwen3_sha256(bytes : BytesView) -> String {
  @crypto.bytes_to_hex_string(@crypto.sha256(bytes))
}

///|
fn qwen3_parse_uint(value : String) -> UInt raise Qwen3PhysicalError {
  if value.is_empty() || value.length() > 6 {
    raise InvalidInput("compiler version component is invalid")
  }
  let mut result = 0U
  for character in value {
    guard character is ('0'..='9') else {
      raise InvalidInput("compiler version component is invalid")
    }
    result = result * 10U +
      (character.to_int() - '0'.to_int()).reinterpret_as_uint()
  }
  result
}

///|
async fn qwen3_read_pinned(
  path : String,
  expected_sha256 : String,
  maximum_bytes : Int,
) -> Bytes raise Qwen3PhysicalError {
  qwen3_require(qwen3_is_sha256(expected_sha256), "digest is not SHA-256")
  let canonical = @fs.realpath(path) catch {
    _ => raise InvalidInput("input path is unavailable")
  }
  qwen3_require(canonical == path, "input path is not canonical")
  let bytes = @fs.read_file(path).binary() catch {
    _ => raise InvalidInput("input file is unreadable")
  }
  qwen3_require(
    bytes.length() > 0 && bytes.length() <= maximum_bytes,
    "input file exceeds its closed size limit",
  )
  if qwen3_sha256(bytes) != expected_sha256 {
    raise IdentityMismatch("input file digest mismatch")
  }
  bytes
}

///|
async fn qwen3_read_canonical(
  path : String,
  maximum_bytes : Int,
) -> Bytes raise Qwen3PhysicalError {
  let canonical = @fs.realpath(path) catch {
    _ => raise InvalidInput("input path is unavailable")
  }
  qwen3_require(canonical == path, "input path is not canonical")
  let bytes = @fs.read_file(path).binary() catch {
    _ => raise InvalidInput("input file is unreadable")
  }
  qwen3_require(
    bytes.length() > 0 && bytes.length() <= maximum_bytes,
    "input file exceeds its closed size limit",
  )
  bytes
}

///|
struct Qwen3PhysicalInputs {
  candidates : @luna_bf16_kernel_producer.LunaBf16CandidateSet
}

///|
struct Qwen3ProbeBalance {
  mut contexts : Int
  mut streams : Int
  mut allocations : Int
  mut modules : Int
  mut functions : Int
  mut cleanup : Int
}

///|
fn qwen3_probe_balance() -> Qwen3ProbeBalance {
  {
    contexts: 0,
    streams: 0,
    allocations: 0,
    modules: 0,
    functions: 0,
    cleanup: 0,
  }
}

///|
fn Qwen3ProbeBalance::is_zero(self : Self) -> Bool {
  self.contexts == 0 &&
  self.streams == 0 &&
  self.allocations == 0 &&
  self.modules == 0 &&
  self.functions == 0 &&
  self.cleanup == 0
}
