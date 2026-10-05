///|
fn parse_positive_uint(
  value : String,
  allow_zero : Bool,
) -> UInt raise Qwen3Bf16ReleaseBindError {
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
fn parse_positive_int(value : String) -> Int raise Qwen3Bf16ReleaseBindError {
  let parsed = parse_positive_uint(value, false)
  if parsed > 2147483647U {
    raise InvalidArguments
  }
  parsed.reinterpret_as_int()
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
fn parse_qwen3_release_bind_arguments(
  arguments : ArrayView[String],
) -> Qwen3ReleaseBindArguments raise Qwen3Bf16ReleaseBindError {
  let mut compute_major = 12
  let mut compute_minor = 0
  let mut target_seen = false
  let mut max_activation_arena_bytes = 2147483648L
  let mut arena_seen = false
  let filtered : Array[String] = []
  for index = 0; index < arguments.length(); {
    if arguments[index] == "--activation-arena-gib" {
      guard !arena_seen && index + 1 < arguments.length() else {
        raise InvalidArguments
      }
      let gib = parse_positive_int(arguments[index + 1])
      guard gib <= 64 else { raise InvalidArguments }
      max_activation_arena_bytes = gib.to_int64() * 1073741824L
      arena_seen = true
      continue index + 2
    }
    if arguments[index] == "--device-target" {
      guard !target_seen && index + 2 < arguments.length() else {
        raise InvalidArguments
      }
      compute_major = parse_positive_int(arguments[index + 1])
      compute_minor = parse_positive_uint(arguments[index + 2], true).reinterpret_as_int()
      guard compute_major >= 8 && compute_minor <= 9 else {
        raise InvalidArguments
      }
      target_seen = true
      continue index + 3
    }
    filtered.push(arguments[index])
    continue index + 1
  }
  let arguments = filtered[:]
  let (arguments, projection_folds) = match arguments {
    [.. rest, "--projection-folds", path, digest, device] => {
      guard path.has_prefix("/") &&
        is_lower_sha256(digest) &&
        !device.is_empty() else {
        raise InvalidArguments
      }
      (rest, Some((path, digest, device)))
    }
    _ => (arguments, None)
  }
  let (arguments, compiler_register_limit) = match arguments {
    [.. rest, "--compiler-register-limit", value] => {
      let limit = parse_positive_int(value)
      guard limit <= 255 else { raise InvalidArguments }
      (rest, limit)
    }
    _ => (arguments, 128)
  }
  guard arguments
    is [
      _,
      model_root,
      config_locator,
      config_sha256,
      model_content_sha256,
      numeric_weights_locator,
      numeric_weight_artifact_sha256,
      weight_route_manifest_sha256,
      compiled_root,
      toolchain_manifest,
      toolchain_sha256,
      compiler_major,
      compiler_minor,
      compiler_patch,
      tokens_per_page,
      total_page_count,
      max_query_rows,
      max_query_tokens,
      max_page_table_entries,
      absolute_new_output,
      .. tuning_suffix,
    ] else {
    raise InvalidArguments
  }
  if !is_lower_sha256(config_sha256) ||
    !is_lower_sha256(model_content_sha256) ||
    !is_lower_sha256(numeric_weight_artifact_sha256) ||
    !is_lower_sha256(weight_route_manifest_sha256) ||
    !is_lower_sha256(toolchain_sha256) {
    raise InvalidArguments
  }
  {
    compute_major,
    compute_minor,
    max_activation_arena_bytes,
    model_root,
    config_locator,
    config_sha256,
    model_content_sha256,
    numeric_weights_locator,
    numeric_weight_artifact_sha256,
    weight_route_manifest_sha256,
    compiled_root,
    toolchain_manifest,
    toolchain_sha256,
    compiler_major: parse_positive_uint(compiler_major, false),
    compiler_minor: parse_positive_uint(compiler_minor, true),
    compiler_patch: parse_positive_uint(compiler_patch, true),
    tokens_per_page: parse_positive_int(tokens_per_page),
    total_page_count: parse_positive_int(total_page_count),
    max_query_rows: parse_positive_int(max_query_rows),
    max_query_tokens: parse_positive_int(max_query_tokens),
    max_page_table_entries: parse_positive_int(max_page_table_entries),
    absolute_new_output,
    projection_folds,
    compiler_register_limit,
    projection_tuning: @projection_tuning_file.parse_projection_tuning_suffix(
      tuning_suffix,
    ) catch {
      _ => raise InvalidArguments
    },
  }
}
