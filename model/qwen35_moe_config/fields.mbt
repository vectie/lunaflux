///|
fn missing(
  area : Qwen35ConfigArea,
  field : Qwen35ConfigField,
) -> Qwen35ConfigError {
  MissingField(area~, field~)
}

///|
fn invalid(
  area : Qwen35ConfigArea,
  field : Qwen35ConfigField,
  issue : Qwen35ConfigValueIssue,
  index? : Int = 0,
) -> Qwen35ConfigError {
  InvalidValue(area~, field~, issue~, index~)
}

///|
fn require_value(
  object : Map[String, Json],
  area : Qwen35ConfigArea,
  name : String,
  field : Qwen35ConfigField,
) -> Json raise Qwen35ConfigError {
  match object.get(name) {
    Some(value) => value
    None => raise missing(area, field)
  }
}

///|
fn require_object(
  object : Map[String, Json],
  area : Qwen35ConfigArea,
  name : String,
  field : Qwen35ConfigField,
) -> Map[String, Json] raise Qwen35ConfigError {
  match require_value(object, area, name, field) {
    Object(value) => value
    _ => raise invalid(area, field, ExpectedObject)
  }
}

///|
fn require_string(
  object : Map[String, Json],
  area : Qwen35ConfigArea,
  name : String,
  field : Qwen35ConfigField,
) -> String raise Qwen35ConfigError {
  match require_value(object, area, name, field) {
    String(value) => value
    _ => raise invalid(area, field, ExpectedString)
  }
}

///|
fn parse_decimal_int64(
  text : StringView,
  area : Qwen35ConfigArea,
  field : Qwen35ConfigField,
  index : Int,
) -> Int64 raise Qwen35ConfigError {
  if text.is_empty() {
    raise invalid(area, field, ExpectedInteger, index~)
  }
  let negative = text[0] == '-'
  let start = if negative { 1 } else { 0 }
  if start == text.length() {
    raise invalid(area, field, ExpectedInteger, index~)
  }
  let limit = if negative { 0x8000000000000000UL } else { 0x7fffffffffffffffUL }
  let mut magnitude = 0UL
  for cursor in start..<text.length() {
    let unit = text[cursor]
    if unit < '0' || unit > '9' {
      raise invalid(area, field, ExpectedInteger, index~)
    }
    let digit = (unit.to_int() - '0').to_uint64()
    if magnitude > (limit - digit) / 10UL {
      raise invalid(area, field, ExpectedInteger, index~)
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
fn exact_int64(
  value : Json,
  area : Qwen35ConfigArea,
  field : Qwen35ConfigField,
  index? : Int = 0,
) -> Int64 raise Qwen35ConfigError {
  guard value is Number(number, repr~) else {
    raise invalid(area, field, ExpectedInteger, index~)
  }
  match repr {
    Some(text) => parse_decimal_int64(text, area, field, index)
    None => {
      if number.is_nan() ||
        number.is_inf() ||
        number != number.trunc() ||
        number < -9007199254740991.0 ||
        number > 9007199254740991.0 {
        raise invalid(area, field, ExpectedInteger, index~)
      }
      number.to_int64()
    }
  }
}

///|
fn require_int(
  object : Map[String, Json],
  area : Qwen35ConfigArea,
  name : String,
  field : Qwen35ConfigField,
) -> Int raise Qwen35ConfigError {
  let value = exact_int64(require_value(object, area, name, field), area, field)
  if value < -2147483648L || value > 2147483647L {
    raise invalid(area, field, ExpectedInteger)
  }
  value.to_int()
}

///|
fn require_number(
  object : Map[String, Json],
  area : Qwen35ConfigArea,
  name : String,
  field : Qwen35ConfigField,
) -> Double raise Qwen35ConfigError {
  match require_value(object, area, name, field) {
    Number(value, repr=_) if !value.is_nan() && !value.is_inf() => value
    _ => raise invalid(area, field, ExpectedNumber)
  }
}

///|
fn require_bool(
  object : Map[String, Json],
  area : Qwen35ConfigArea,
  name : String,
  field : Qwen35ConfigField,
) -> Bool raise Qwen35ConfigError {
  match require_value(object, area, name, field) {
    True => true
    False => false
    _ => raise invalid(area, field, ExpectedBoolean)
  }
}

///|
fn require_null(
  object : Map[String, Json],
  area : Qwen35ConfigArea,
  name : String,
  field : Qwen35ConfigField,
) -> Unit raise Qwen35ConfigError {
  match require_value(object, area, name, field) {
    Null => ()
    _ => raise invalid(area, field, ExpectedNull)
  }
}

///|
fn expect_string(
  object : Map[String, Json],
  area : Qwen35ConfigArea,
  name : String,
  field : Qwen35ConfigField,
  expected : String,
) -> Unit raise Qwen35ConfigError {
  if require_string(object, area, name, field) != expected {
    raise invalid(area, field, UnsupportedValue)
  }
}

///|
fn expect_int(
  object : Map[String, Json],
  area : Qwen35ConfigArea,
  name : String,
  field : Qwen35ConfigField,
  expected : Int,
) -> Unit raise Qwen35ConfigError {
  if require_int(object, area, name, field) != expected {
    raise invalid(area, field, UnsupportedValue)
  }
}

///|
fn expect_number(
  object : Map[String, Json],
  area : Qwen35ConfigArea,
  name : String,
  field : Qwen35ConfigField,
  expected : Double,
) -> Unit raise Qwen35ConfigError {
  if require_number(object, area, name, field) != expected {
    raise invalid(area, field, UnsupportedValue)
  }
}

///|
fn expect_bool(
  object : Map[String, Json],
  area : Qwen35ConfigArea,
  name : String,
  field : Qwen35ConfigField,
  expected : Bool,
) -> Unit raise Qwen35ConfigError {
  if require_bool(object, area, name, field) != expected {
    raise invalid(area, field, UnsupportedValue)
  }
}
