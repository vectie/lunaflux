///|
const APPROVED_PATH_MAX_BYTES : Int = 4096

///|
fn validate_segments(
  value : StringView,
  start : Int,
  rule : ApprovedFsRule,
) -> Unit raise ApprovedFsError {
  let mut segment_length = 0
  let mut segment_dots = 0
  for index in start..<value.length() {
    let unit = value[index]
    if unit == '/' {
      if segment_length == 0 ||
        (segment_length <= 2 && segment_dots == segment_length) {
        raise Invalid(rule)
      }
      segment_length = 0
      segment_dots = 0
    } else {
      let valid = unit is ('A'..='Z' | 'a'..='z' | '0'..='9' | '_' | '-' | '.')
      if !valid {
        raise Invalid(rule)
      }
      segment_length += 1
      if unit == '.' {
        segment_dots += 1
      }
    }
  }
  if segment_length == 0 ||
    (segment_length <= 2 && segment_dots == segment_length) {
    raise Invalid(rule)
  }
}

///|
fn validate_root_path(value : StringView) -> Unit raise ApprovedFsError {
  if value.length() < 2 ||
    value.length() > APPROVED_PATH_MAX_BYTES ||
    value[0] != '/' {
    raise Invalid(RootPath)
  }
  validate_segments(value, 1, RootPath)
}

///|
fn validate_relative_locator(value : StringView) -> Unit raise ApprovedFsError {
  if value.is_empty() ||
    value.length() > APPROVED_PATH_MAX_BYTES ||
    value[0] == '/' {
    raise Invalid(RelativeLocator)
  }
  validate_segments(value, 0, RelativeLocator)
}
