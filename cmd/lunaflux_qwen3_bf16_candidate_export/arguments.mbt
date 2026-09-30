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
  let mut compute_major = 12
  let mut compute_minor = 0
  let mut target_seen = false
  let filtered : Array[String] = []
  for index = 0; index < arguments.length(); {
    if arguments[index] == "--device-target" {
      guard !target_seen && index + 2 < arguments.length() else {
        raise InvalidArguments
      }
      compute_major = parse_positive_decimal(arguments[index + 1])
      compute_minor = parse_compiler_component(arguments[index + 2]).reinterpret_as_int()
      guard compute_major >= 8 && compute_minor <= 9 else {
        raise InvalidArguments
      }
      target_seen = true
      continue index + 3
    }
    filtered.push(arguments[index])
    continue index + 1
  }
  let arguments = filtered
  let flags = arguments.filter(value => value == "--ingress-policy-search")
  guard flags.length() <= 1 else { raise InvalidArguments }
  let ingress_policy_search = !flags.is_empty()
  let arguments = arguments.filter(value => value != "--ingress-policy-search")
  let limit_flags = arguments.filter(value => value == "--ingress-shared-limit")
  guard limit_flags.length() <= 1 else { raise InvalidArguments }
  let mut ingress_shared_limit = 49152L
  let filtered : Array[String] = []
  for index = 0; index < arguments.length(); {
    if arguments[index] == "--ingress-shared-limit" {
      guard index + 1 < arguments.length() else { raise InvalidArguments }
      let bytes = parse_positive_decimal(arguments[index + 1])
      guard bytes <= 98304 else { raise InvalidArguments }
      ingress_shared_limit = bytes.to_int64()
      continue index + 2
    }
    filtered.push(arguments[index])
    continue index + 1
  }
  let arguments = filtered
  let mut prefill_candidate : Int? = None
  let mut wide_prefill_candidate : Int? = None
  let filtered : Array[String] = []
  for index = 0; index < arguments.length(); {
    if arguments[index] == "--prefill-candidate" ||
      arguments[index] == "--wide-prefill-candidate" {
      guard index + 1 < arguments.length() else { raise InvalidArguments }
      let candidate = parse_positive_decimal(arguments[index + 1])
      if arguments[index] == "--prefill-candidate" {
        guard prefill_candidate is None else { raise InvalidArguments }
        prefill_candidate = Some(candidate)
      } else {
        guard wide_prefill_candidate is None else { raise InvalidArguments }
        wide_prefill_candidate = Some(candidate)
      }
      continue index + 2
    }
    filtered.push(arguments[index])
    continue index + 1
  }
  let arguments = filtered
  let mut compiler_register_limit = 128
  let mut register_limit_seen = false
  let filtered : Array[String] = []
  for index = 0; index < arguments.length(); {
    if arguments[index] == "--compiler-register-limit" {
      guard !register_limit_seen && index + 1 < arguments.length() else {
        raise InvalidArguments
      }
      let limit = parse_positive_decimal(arguments[index + 1])
      guard limit <= 255 else { raise InvalidArguments }
      compiler_register_limit = limit
      register_limit_seen = true
      continue index + 2
    }
    filtered.push(arguments[index])
    continue index + 1
  }
  let arguments = filtered
  let mut decode_candidate : Int? = None
  let filtered : Array[String] = []
  for index = 0; index < arguments.length(); {
    if arguments[index] == "--decode-candidate" {
      guard decode_candidate is None && index + 1 < arguments.length() else {
        raise InvalidArguments
      }
      decode_candidate = Some(parse_positive_decimal(arguments[index + 1]))
      continue index + 2
    }
    filtered.push(arguments[index])
    continue index + 1
  }
  let arguments = filtered
  let (arguments, ingress_policy_records) = extract_resource_option(
    arguments[:],
    "--ingress-policy-records",
  )
  let (arguments, decode_resources) = extract_resource_option(
    arguments[:],
    "--decode-resources",
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
    compute_major,
    compute_minor,
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
    ingress_policy_records,
    ingress_policy_search,
    ingress_shared_limit,
    compiler_register_limit,
    prefill_candidate,
    wide_prefill_candidate,
    decode_candidate,
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
      decode_tuning, partitioned_resources, ingress_policy_records,
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
