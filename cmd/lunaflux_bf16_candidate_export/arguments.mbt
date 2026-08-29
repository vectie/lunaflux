///|
fn parse_positive_decimal(
  value : String,
) -> Int raise LunaBf16CandidateExportCliError {
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
) -> UInt raise LunaBf16CandidateExportCliError {
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
fn parse_candidate_export_arguments(
  arguments : ArrayView[String],
) -> CandidateExportArguments raise LunaBf16CandidateExportCliError {
  guard arguments
    is [
      _,
      model_root,
      config_locator,
      config_sha256,
      model_content_sha256,
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
  let parsed : CandidateExportArguments = {
    model_root,
    config_locator,
    config_sha256,
    model_content_sha256,
    toolchain_sha256,
    compiler_major: parse_compiler_component(compiler_major),
    compiler_minor: parse_compiler_component(compiler_minor),
    compiler_patch: parse_compiler_component(compiler_patch),
    tokens_per_page: parse_positive_decimal(tokens_per_page),
    total_page_count: parse_positive_decimal(total_page_count),
    max_query_tokens: parse_positive_decimal(max_query_tokens),
    max_page_table_entries: parse_positive_decimal(max_page_table_entries),
    absolute_new_output,
  }
  if parsed.compiler_major == 0U {
    raise InvalidArguments
  }
  parsed
}
