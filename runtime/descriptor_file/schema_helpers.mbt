///|
fn invalid_field(
  area : RuntimeDescriptorArea,
  issue : RuntimeDescriptorIssue,
  index? : Int = 0,
) -> RuntimeDescriptorFileError {
  InvalidField(area~, issue~, index~)
}

///|
fn validate_fields(
  object : Map[String, Json],
  area : RuntimeDescriptorArea,
  allowed : ArrayView[String],
) -> Unit raise RuntimeDescriptorFileError {
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
fn validate_required_and_optional_fields(
  object : Map[String, Json],
  area : RuntimeDescriptorArea,
  required : ArrayView[String],
  optional : ArrayView[String],
) -> Unit raise RuntimeDescriptorFileError {
  let mut field_index = 0
  for name, _ in object {
    if !required.contains(name) && !optional.contains(name) {
      raise UnknownField(area~, index=field_index)
    }
    field_index += 1
  }
  for index, name in required {
    if !object.contains(name) {
      raise MissingField(area~, index~)
    }
  }
}

///|
fn require_object(
  object : Map[String, Json],
  name : String,
  area : RuntimeDescriptorArea,
  index? : Int = 0,
) -> Map[String, Json] raise RuntimeDescriptorFileError {
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
  area : RuntimeDescriptorArea,
  index? : Int = 0,
) -> String raise RuntimeDescriptorFileError {
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
  area : RuntimeDescriptorArea,
  index? : Int = 0,
) -> Bool raise RuntimeDescriptorFileError {
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
  area : RuntimeDescriptorArea,
  index : Int,
) -> UInt64 raise RuntimeDescriptorFileError {
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
fn parse_uint64(
  value : Json,
  area : RuntimeDescriptorArea,
  index : Int,
) -> UInt64 raise RuntimeDescriptorFileError {
  guard value is Number(number, repr~) else {
    raise invalid_field(area, ExpectedInteger, index~)
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
fn require_uint64(
  object : Map[String, Json],
  name : String,
  area : RuntimeDescriptorArea,
  index? : Int = 0,
) -> UInt64 raise RuntimeDescriptorFileError {
  match object.get(name) {
    Some(value) => parse_uint64(value, area, index)
    None => raise MissingField(area~, index~)
  }
}

///|
fn require_int64(
  object : Map[String, Json],
  name : String,
  area : RuntimeDescriptorArea,
  index? : Int = 0,
) -> Int64 raise RuntimeDescriptorFileError {
  let value = require_uint64(object, name, area, index~)
  if value > 0x7fffffffffffffffUL {
    raise invalid_field(area, UnsupportedValue, index~)
  }
  value.reinterpret_as_int64()
}

///|
fn require_int(
  object : Map[String, Json],
  name : String,
  area : RuntimeDescriptorArea,
  index? : Int = 0,
) -> Int raise RuntimeDescriptorFileError {
  let value = require_uint64(object, name, area, index~)
  if value > 2147483647UL {
    raise invalid_field(area, UnsupportedValue, index~)
  }
  value.to_int()
}

///|
fn require_number(
  object : Map[String, Json],
  name : String,
  area : RuntimeDescriptorArea,
  index? : Int = 0,
) -> Double raise RuntimeDescriptorFileError {
  match object.get(name) {
    Some(Number(value, repr=_)) if !value.is_nan() && !value.is_inf() => value
    Some(Number(_, repr=_)) =>
      raise invalid_field(area, UnsupportedValue, index~)
    Some(_) => raise invalid_field(area, ExpectedNumber, index~)
    None => raise MissingField(area~, index~)
  }
}
