///|
fn validate_fields(
  object : Map[String, Json],
  area : LaunchFileArea,
  expected : ArrayView[String],
) -> Unit raise LaunchFileError {
  for name, _ in object {
    if !expected.contains(name) {
      raise UnknownField(area)
    }
  }
  for name in expected {
    if !object.contains(name) {
      raise MissingField(area)
    }
  }
}

///|
fn require_object(
  object : Map[String, Json],
  name : String,
  area : LaunchFileArea,
) -> Map[String, Json] raise LaunchFileError {
  match object.get(name) {
    Some(Object(value)) => value
    Some(_) => raise InvalidField(area)
    None => raise MissingField(area)
  }
}

///|
fn require_string(
  object : Map[String, Json],
  name : String,
  area : LaunchFileArea,
) -> String raise LaunchFileError {
  match object.get(name) {
    Some(String(value)) => value
    Some(_) => raise InvalidField(area)
    None => raise MissingField(area)
  }
}

///|
fn require_sha256(
  object : Map[String, Json],
  name : String,
  area : LaunchFileArea,
) -> String raise LaunchFileError {
  let value = require_string(object, name, area)
  if !is_lower_sha256(value) {
    raise InvalidField(area)
  }
  value
}

///|
/// Mirrors the strict approved-root lexical grammar without opening it.
fn validate_absolute_label(value : StringView) -> Unit raise LaunchFileError {
  if value.length() < 2 || value.length() > 4096 || value[0] != '/' {
    raise InvalidField(Root)
  }
  let mut segment_length = 0
  let mut dots = 0
  for index in 1..<value.length() {
    let unit = value[index]
    if unit == '/' {
      if segment_length == 0 || (segment_length <= 2 && dots == segment_length) {
        raise InvalidField(Root)
      }
      segment_length = 0
      dots = 0
    } else {
      if !(unit is ('A'..='Z' | 'a'..='z' | '0'..='9' | '_' | '-' | '.')) {
        raise InvalidField(Root)
      }
      segment_length += 1
      if unit == '.' {
        dots += 1
      }
    }
  }
  if segment_length == 0 || (segment_length <= 2 && dots == segment_length) {
    raise InvalidField(Root)
  }
}

///|
fn validate_locator(
  value : StringView,
  area : LaunchFileArea,
) -> Unit raise LaunchFileError {
  ignore(
    @approved_fs.ApprovedRelativeLocator::new(value) catch {
      _ => raise InvalidField(area)
    },
  )
}
