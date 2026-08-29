///|
fn canonical_sha256(value : StringView) -> Bool {
  value.length() == 64 &&
  value.all(ch => ch.is_ascii_digit() || (ch >= 'a' && ch <= 'f'))
}

///|
fn parse_qwen3_weight_convert_arguments(
  arguments : ArrayView[String],
) -> Qwen3WeightConvertArguments raise Qwen3WeightConvertError {
  guard arguments
    is [
      _,
      model_root,
      config_locator,
      config_sha256,
      model_content_sha256,
      source_locator,
      source_sha256,
      output_locator,
    ] else {
    raise InvalidArguments
  }
  if !canonical_sha256(config_sha256) ||
    !canonical_sha256(model_content_sha256) ||
    !canonical_sha256(source_sha256) ||
    config_locator.contains("/") ||
    source_locator.contains("/") ||
    output_locator.contains("/") ||
    source_locator == output_locator {
    raise InvalidArguments
  }
  ignore(
    @approved_fs.ApprovedRelativeLocator::new(config_locator) catch {
      _ => raise InvalidArguments
    },
  )
  ignore(
    @approved_fs.ApprovedRelativeLocator::new(source_locator) catch {
      _ => raise InvalidArguments
    },
  )
  ignore(
    @approved_fs.ApprovedRelativeLocator::new(output_locator) catch {
      _ => raise InvalidArguments
    },
  )
  {
    model_root,
    config_locator,
    config_sha256,
    model_content_sha256,
    source_locator,
    source_sha256,
    output_locator,
  }
}
