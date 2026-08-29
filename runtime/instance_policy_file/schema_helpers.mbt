///|
fn invalid_field(
  area : InstancePolicyArea,
  issue : InstancePolicyIssue,
  index? : Int = 0,
) -> InstancePolicyFileError {
  InvalidField(area~, issue~, index~)
}

///|
fn validate_fields(
  object : Map[String, Json],
  area : InstancePolicyArea,
  allowed : ArrayView[String],
) -> Unit raise InstancePolicyFileError {
  let mut field_index = 0
  for name, _ in object {
    if !allowed.contains(name) {
      raise UnknownField(area~, index=field_index)
    }
    field_index += 1
  }
  for index, name in allowed {
    if !object.contains(name) {
      raise MissingField(area~, index~)
    }
  }
}

///|
fn require_object(
  object : Map[String, Json],
  name : String,
  area : InstancePolicyArea,
  index? : Int = 0,
) -> Map[String, Json] raise InstancePolicyFileError {
  match object.get(name) {
    Some(Object(value)) => value
    Some(_) => raise invalid_field(area, ExpectedObject, index~)
    None => raise MissingField(area~, index~)
  }
}

///|
fn require_string(
  object : Map[String, Json],
  name : String,
  area : InstancePolicyArea,
  index? : Int = 0,
) -> String raise InstancePolicyFileError {
  match object.get(name) {
    Some(String(value)) => value
    Some(_) => raise invalid_field(area, ExpectedString, index~)
    None => raise MissingField(area~, index~)
  }
}

///|
fn require_bool(
  object : Map[String, Json],
  name : String,
  area : InstancePolicyArea,
  index? : Int = 0,
) -> Bool raise InstancePolicyFileError {
  match object.get(name) {
    Some(True) => true
    Some(False) => false
    Some(_) => raise invalid_field(area, ExpectedBoolean, index~)
    None => raise MissingField(area~, index~)
  }
}

///|
fn parse_uint64_text(
  text : StringView,
  area : InstancePolicyArea,
  index : Int,
) -> UInt64 raise InstancePolicyFileError {
  if text.is_empty() {
    raise invalid_field(area, ExpectedInteger, index~)
  }
  let mut result = 0UL
  for unit in text {
    if !(unit is ('0'..='9')) {
      raise invalid_field(area, ExpectedInteger, index~)
    }
    let digit = (unit.to_int() - '0'.to_int()).to_uint64()
    if result > (0xffffffffffffffffUL - digit) / 10UL {
      raise invalid_field(area, UnsupportedValue, index~)
    }
    result = result * 10UL + digit
  }
  result
}

///|
fn require_uint64(
  object : Map[String, Json],
  name : String,
  area : InstancePolicyArea,
  index? : Int = 0,
) -> UInt64 raise InstancePolicyFileError {
  guard object.get(name) is Some(Number(number, repr~)) else {
    if object.contains(name) {
      raise invalid_field(area, ExpectedInteger, index~)
    }
    raise MissingField(area~, index~)
  }
  match repr {
    Some(text) => parse_uint64_text(text, area, index)
    None => {
      if number.is_nan() ||
        number.is_inf() ||
        number != number.trunc() ||
        number < 0.0 ||
        number > 9007199254740991.0 {
        raise invalid_field(area, ExpectedInteger, index~)
      }
      number.to_uint64()
    }
  }
}

///|
fn require_int(
  object : Map[String, Json],
  name : String,
  area : InstancePolicyArea,
  index? : Int = 0,
) -> Int raise InstancePolicyFileError {
  let value = require_uint64(object, name, area, index~)
  if value > 2147483647UL {
    raise invalid_field(area, UnsupportedValue, index~)
  }
  value.to_int()
}

///|
fn require_lower_sha256(
  object : Map[String, Json],
  name : String,
  area : InstancePolicyArea,
  index : Int,
) -> String raise InstancePolicyFileError {
  let value = require_string(object, name, area, index~)
  if value.length() != 64 {
    raise invalid_field(area, UnsupportedValue, index~)
  }
  for unit in value {
    if !(unit is ('0'..='9' | 'a'..='f')) {
      raise invalid_field(area, UnsupportedValue, index~)
    }
  }
  value
}

///|
fn require_relative_locator(
  object : Map[String, Json],
  name : String,
  area : InstancePolicyArea,
  index : Int,
) -> String raise InstancePolicyFileError {
  let value = require_string(object, name, area, index~)
  ignore(
    @approved_fs.ApprovedRelativeLocator::new(value) catch {
      _ => raise invalid_field(area, UnsupportedValue, index~)
    },
  )
  value
}
