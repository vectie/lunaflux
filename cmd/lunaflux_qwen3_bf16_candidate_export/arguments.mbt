///|
fn parse_positive_decimal(
  value : String,
) -> Int raise Qwen3Bf16CandidateExportCliError {
  if value.is_empty() || value.length() > 9 {
    raise InvalidArguments
  }
  let mut result = 0
  for unit in value {
    guard unit is ('0'..='9') else { raise InvalidArguments }
    let digit = unit.to_int() - '0'.to_int()
    if result > (2147483647 - digit) / 10 {
      raise InvalidArguments
    }
    result = result * 10 + digit
  }
  if result <= 0 {
    raise InvalidArguments
  }
  result
}

///|
fn parse_compiler_component(
  value : String,
) -> UInt raise Qwen3Bf16CandidateExportCliError {
  if value.is_empty() || value.length() > 6 {
    raise InvalidArguments
  }
  let mut result = 0U
  for unit in value {
    guard unit is ('0'..='9') else { raise InvalidArguments }
    result = result * 10U + (unit.to_int() - '0'.to_int()).reinterpret_as_uint()
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
fn parse_qwen3_candidate_export_arguments(
  arguments : ArrayView[String],
) -> Qwen3CandidateExportArguments raise Qwen3Bf16CandidateExportCliError {
  let (arguments, decode_resources) = extract_resource_option(
    arguments, "--decode-resources",
  )
  let (arguments, decode_tuning) = extract_resource_option(
    arguments[:],
    "--decode-tuning",
  )
  let (arguments, partitioned_resources) = extract_resource_option(
    arguments[:],
    "--partitioned-resources",
  )
  let arguments = arguments[:]
  let (arguments, query_metadata) = match arguments {
    [.. rest, "--query-metadata-v1"] => (rest, true)
    _ => (arguments, false)
  }
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
  let (arguments, attention_resources) = match arguments {
    [.. rest, "--attention-resources", path, digest, device] => {
      guard path.has_prefix("/") &&
        is_lower_sha256(digest) &&
        !device.is_empty() else {
        raise InvalidArguments
      }
      (rest, Some((path, digest, device)))
    }
    _ => (arguments, None)
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
  let (projection_suffix, attention_tuning) = match tuning_suffix {
    [.. projection, "--attention-tuning", path, digest, device] => {
      if !path.has_prefix("/") || !is_lower_sha256(digest) || device.is_empty() {
        raise InvalidArguments
      }
      (projection, Some((path, digest, device)))
    }
    _ => (tuning_suffix, None)
  }
  let parsed : Qwen3CandidateExportArguments = {
    model_root,
    config_locator,
    config_sha256,
    model_content_sha256,
    numeric_weights_locator,
    numeric_weight_artifact_sha256,
    weight_route_manifest_sha256,
    toolchain_sha256,
    compiler_major: parse_compiler_component(compiler_major),
    compiler_minor: parse_compiler_component(compiler_minor),
    compiler_patch: parse_compiler_component(compiler_patch),
    tokens_per_page: parse_positive_decimal(tokens_per_page),
    total_page_count: parse_positive_decimal(total_page_count),
    max_query_rows: parse_positive_decimal(max_query_rows),
    max_query_tokens: parse_positive_decimal(max_query_tokens),
    max_page_table_entries: parse_positive_decimal(max_page_table_entries),
    absolute_new_output,
    projection_tuning: @projection_tuning_file.parse_projection_tuning_suffix(
      projection_suffix,
    ) catch {
      _ => raise InvalidArguments
    },
    attention_tuning,
    attention_resources,
    projection_folds,
    decode_resources,
    decode_tuning,
    partitioned_resources,
    query_metadata,
  }
  if attention_tuning is Some((_, _, tuning_device)) &&
    attention_resources is Some((_, _, resource_device)) &&
    tuning_device != resource_device {
    raise InvalidArguments
  }
  if parsed.compiler_major == 0U {
    raise InvalidArguments
  }
  for
    input in [
      attention_tuning, attention_resources, projection_folds, decode_resources,
      decode_tuning, partitioned_resources,
    ]
    device = None {
    guard input is Some((_, _, current)) else { continue device }
    if device is Some(previous) && previous != current {
      raise InvalidArguments
    }
    continue Some(current)
  }
  parsed
}

///|
fn extract_resource_option(
  arguments : ArrayView[String],
  key : String,
) -> (ReadOnlyArray[String], (String, String, String)?) raise Qwen3Bf16CandidateExportCliError {
  let remaining : Array[String] = []
  for index = 0, found = None; index < arguments.length(); {
    if arguments[index] != key {
      remaining.push(arguments[index])
      continue index + 1, found
    }
    guard found is None &&
      arguments[index + 1:] is [path, digest, device, ..] &&
      path.has_prefix("/") &&
      is_lower_sha256(digest) &&
      !device.is_empty() else {
      raise InvalidArguments
    }
    continue index + 4, Some((path, digest, device))
  } nobreak {
    (ReadOnlyArray::from_array(remaining), found)
  }
}
