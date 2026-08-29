///|
fn parse_positive_uint(
  value : String,
  allow_zero : Bool,
) -> UInt raise LunaBf16ReleaseBindError {
  if value.is_empty() || value.length() > 6 {
    raise InvalidArguments
  }
  let mut result = 0U
  for unit in value {
    guard unit is ('0'..='9') else { raise InvalidArguments }
    result = result * 10U + (unit.to_int() - '0'.to_int()).reinterpret_as_uint()
  }
  if result == 0U && !allow_zero {
    raise InvalidArguments
  }
  result
}

///|
fn is_lower_sha256(value : StringView) -> Bool {
  if value.length() != 64 {
    return false
  }
  for unit in value {
    if !(unit is ('0'..='9' | 'a'..='f')) {
      return false
    }
  }
  true
}

///|
fn parse_release_bind_arguments(
  arguments : ArrayView[String],
) -> ReleaseBindArguments raise LunaBf16ReleaseBindError {
  guard arguments
    is [
      _,
      model_root,
      compiled_root,
      toolchain_manifest,
      toolchain_sha256,
      compiler_major,
      compiler_minor,
      compiler_patch,
      absolute_new_output,
    ] else {
    raise InvalidArguments
  }
  if !is_lower_sha256(toolchain_sha256) {
    raise InvalidArguments
  }
  {
    model_root,
    compiled_root,
    toolchain_manifest,
    toolchain_sha256,
    compiler_major: parse_positive_uint(compiler_major, false),
    compiler_minor: parse_positive_uint(compiler_minor, true),
    compiler_patch: parse_positive_uint(compiler_patch, true),
    absolute_new_output,
  }
}
