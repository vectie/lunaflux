///|
fn missing(field : ModelConfigField) -> ModelConfigReaderError {
  MissingField(field)
}

///|
fn invalid(
  field : ModelConfigField,
  issue : ModelConfigValueIssue,
  index? : Int = 0,
) -> ModelConfigReaderError {
  InvalidValue(field~, issue~, index~)
}

///|
fn require_value(
  object : Map[String, Json],
  name : String,
  field : ModelConfigField,
) -> Json raise ModelConfigReaderError {
  match object.get(name) {
    Some(value) => value
    None => raise missing(field)
  }
}

///|
fn require_string(
  object : Map[String, Json],
  name : String,
  field : ModelConfigField,
) -> String raise ModelConfigReaderError {
  match require_value(object, name, field) {
    String(value) => value
    _ => raise invalid(field, ExpectedString)
  }
}

///|
fn exact_int64(
  value : Json,
  field : ModelConfigField,
  index? : Int = 0,
) -> Int64 raise ModelConfigReaderError {
  guard value is Number(number, repr~) else {
    raise invalid(field, ExpectedInteger, index~)
  }
  match repr {
    Some(text) => parse_decimal_int64(text, field, index)
    None => {
      if number.is_nan() ||
        number.is_inf() ||
        number != number.trunc() ||
        number < -9007199254740991.0 ||
        number > 9007199254740991.0 {
        raise invalid(field, ExpectedInteger, index~)
      }
      number.to_int64()
    }
  }
}

///|
fn parse_decimal_int64(
  text : StringView,
  field : ModelConfigField,
  index : Int,
) -> Int64 raise ModelConfigReaderError {
  if text.is_empty() {
    raise invalid(field, ExpectedInteger, index~)
  }
  let negative = text[0] == '-'
  let start = if negative { 1 } else { 0 }
  if start == text.length() {
    raise invalid(field, ExpectedInteger, index~)
  }
  let limit = if negative { 0x8000000000000000UL } else { 0x7fffffffffffffffUL }
  let mut magnitude = 0UL
  for cursor in start..<text.length() {
    let unit = text[cursor]
    if unit < '0' || unit > '9' {
      raise invalid(field, ExpectedInteger, index~)
    }
    let digit = (unit.to_int() - '0').to_uint64()
    if magnitude > (limit - digit) / 10UL {
      raise invalid(field, ExpectedInteger, index~)
    }
    magnitude = magnitude * 10UL + digit
  }
  if negative {
    if magnitude == 0x8000000000000000UL {
      magnitude.reinterpret_as_int64()
    } else {
      -magnitude.reinterpret_as_int64()
    }
  } else {
    magnitude.reinterpret_as_int64()
  }
}

///|
fn require_int(
  object : Map[String, Json],
  name : String,
  field : ModelConfigField,
) -> Int raise ModelConfigReaderError {
  let integer = exact_int64(require_value(object, name, field), field)
  if integer < -2147483648L || integer > 2147483647L {
    raise invalid(field, ExpectedInteger)
  }
  integer.to_int()
}

///|
fn optional_int(
  object : Map[String, Json],
  name : String,
  field : ModelConfigField,
) -> Int? raise ModelConfigReaderError {
  match object.get(name) {
    None => None
    Some(value) => {
      let integer = exact_int64(value, field)
      if integer < -2147483648L || integer > 2147483647L {
        raise invalid(field, ExpectedInteger)
      }
      Some(integer.to_int())
    }
  }
}

///|
fn require_number(
  object : Map[String, Json],
  name : String,
  field : ModelConfigField,
) -> Double raise ModelConfigReaderError {
  match require_value(object, name, field) {
    Number(value, repr=_) if !value.is_nan() && !value.is_inf() => value
    _ => raise invalid(field, ExpectedNumber)
  }
}

///|
fn optional_finite_number(
  object : Map[String, Json],
  name : String,
  field : ModelConfigField,
  default : Double,
) -> Double raise ModelConfigReaderError {
  match object.get(name) {
    None => default
    Some(Number(value, repr=_)) if !value.is_nan() && !value.is_inf() => value
    Some(Number(_, repr=_)) => raise invalid(field, UnsupportedValue)
    Some(_) => raise invalid(field, ExpectedNumber)
  }
}

///|
fn require_bool(
  object : Map[String, Json],
  name : String,
  field : ModelConfigField,
) -> Bool raise ModelConfigReaderError {
  match require_value(object, name, field) {
    True => true
    False => false
    _ => raise invalid(field, ExpectedBoolean)
  }
}

///|
fn require_single_architecture(
  object : Map[String, Json],
  expected : String,
) -> Unit raise ModelConfigReaderError {
  match require_value(object, "architectures", Architectures) {
    Array(values) => {
      if values.length() != 1 {
        raise invalid(Architectures, UnsupportedValue)
      }
      match values[0] {
        String(value) if value == expected => ()
        String(_) => raise invalid(Architectures, UnsupportedValue)
        _ => raise invalid(Architectures, InvalidArrayElement)
      }
    }
    _ => raise invalid(Architectures, ExpectedArray)
  }
}

///|
fn optional_nullable_int(
  object : Map[String, Json],
  name : String,
  field : ModelConfigField,
) -> Int? raise ModelConfigReaderError {
  match object.get(name) {
    None | Some(Null) => None
    Some(value) => {
      let integer = exact_int64(value, field)
      if integer < -2147483648L || integer > 2147483647L {
        raise invalid(field, ExpectedInteger)
      }
      Some(integer.to_int())
    }
  }
}

///|
fn check_optional_null(
  object : Map[String, Json],
  name : String,
  field : ModelConfigField,
) -> Unit raise ModelConfigReaderError {
  match object.get(name) {
    None | Some(Null) => ()
    Some(_) => raise invalid(field, ExpectedNull)
  }
}

///|
fn check_optional_string(
  object : Map[String, Json],
  name : String,
  field : ModelConfigField,
  expected : String,
) -> Unit raise ModelConfigReaderError {
  match object.get(name) {
    None => ()
    Some(String(value)) if value == expected => ()
    Some(String(_)) => raise invalid(field, UnsupportedValue)
    Some(_) => raise invalid(field, ExpectedString)
  }
}

///|
fn check_optional_bool(
  object : Map[String, Json],
  name : String,
  field : ModelConfigField,
  expected : Bool,
) -> Unit raise ModelConfigReaderError {
  match object.get(name) {
    None => ()
    Some(True) if expected => ()
    Some(False) if !expected => ()
    Some(True | False) => raise invalid(field, UnsupportedValue)
    Some(_) => raise invalid(field, ExpectedBoolean)
  }
}

///|
fn check_optional_number(
  object : Map[String, Json],
  name : String,
  field : ModelConfigField,
  expected : Double,
) -> Unit raise ModelConfigReaderError {
  match object.get(name) {
    None => ()
    Some(Number(value, repr=_)) if value == expected => ()
    Some(Number(_, repr=_)) => raise invalid(field, UnsupportedValue)
    Some(_) => raise invalid(field, ExpectedNumber)
  }
}

///|
fn check_optional_int(
  object : Map[String, Json],
  name : String,
  field : ModelConfigField,
  expected : Int,
) -> Unit raise ModelConfigReaderError {
  match optional_int(object, name, field) {
    None => ()
    Some(value) if value == expected => ()
    Some(_) => raise invalid(field, UnsupportedValue)
  }
}
