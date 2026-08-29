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
      max_query_tokens,
      max_page_table_entries,
      absolute_new_output,
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
    max_query_tokens: parse_positive_int(max_query_tokens),
    max_page_table_entries: parse_positive_int(max_page_table_entries),
    absolute_new_output,
  }
}
