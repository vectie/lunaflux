///|
fn validate_fields(
  object : Map[String, Json],
  area : DocumentArea,
  allowed : ArrayView[(String, DocumentField)],
) -> Unit raise ConfigDocumentError {
  let mut index = 0
  for name, _ in object {
    let mut known = false
    for _, entry in allowed {
      if name == entry.0 {
        known = true
        break
      }
    }
    if !known {
      raise UnknownField(area~, index~)
    }
    index += 1
  }
  for _, entry in allowed {
    if !object.contains(entry.0) {
      raise MissingField(area~, field=entry.1)
    }
  }
}

///|
fn require_string(
  object : Map[String, Json],
  area : DocumentArea,
  name : String,
  field : DocumentField,
) -> String raise ConfigDocumentError {
  match object.get(name) {
    Some(String(value)) => value
    Some(_) => raise InvalidValue(area~, field~, issue=ExpectedString, index=0)
    None => raise MissingField(area~, field~)
  }
}

///|
fn require_int(
  object : Map[String, Json],
  area : DocumentArea,
  name : String,
  field : DocumentField,
) -> Int raise ConfigDocumentError {
  match object.get(name) {
    Some(value) => {
      let integer = exact_int64(value, area, field, 0)
      if integer < -2147483648L || integer > 2147483647L {
        raise InvalidValue(
          area~,
          field~,
          issue=RejectedByFocusedValidator,
          index=0,
        )
      }
      integer.to_int()
    }
    None => raise MissingField(area~, field~)
  }
}

///|
fn require_int64(
  object : Map[String, Json],
  area : DocumentArea,
  name : String,
  field : DocumentField,
) -> Int64 raise ConfigDocumentError {
  match object.get(name) {
    Some(value) => exact_int64(value, area, field, 0)
    None => raise MissingField(area~, field~)
  }
}

///|
fn require_string_array(
  object : Map[String, Json],
  area : DocumentArea,
  name : String,
  field : DocumentField,
) -> Array[String] raise ConfigDocumentError {
  match object.get(name) {
    Some(Array(values)) => {
      let result : Array[String] = []
      for index, value in values {
        match value {
          String(text) => result.push(text)
          _ =>
            raise InvalidValue(area~, field~, issue=InvalidArrayElement, index~)
        }
      }
      result
    }
    Some(_) => raise InvalidValue(area~, field~, issue=ExpectedArray, index=0)
    None => raise MissingField(area~, field~)
  }
}

///|
fn require_int_array(
  object : Map[String, Json],
  area : DocumentArea,
  name : String,
  field : DocumentField,
) -> Array[Int] raise ConfigDocumentError {
  match object.get(name) {
    Some(Array(values)) => {
      let result : Array[Int] = []
      for index, value in values {
        let integer = exact_int64(value, area, field, index)
        if integer < -2147483648L || integer > 2147483647L {
          raise InvalidValue(area~, field~, issue=InvalidArrayElement, index~)
        }
        result.push(integer.to_int())
      }
      result
    }
    Some(_) => raise InvalidValue(area~, field~, issue=ExpectedArray, index=0)
    None => raise MissingField(area~, field~)
  }
}

///|
fn exact_int64(
  value : Json,
  area : DocumentArea,
  field : DocumentField,
  index : Int,
) -> Int64 raise ConfigDocumentError {
  guard value is Number(number, repr~) else {
    raise InvalidValue(area~, field~, issue=ExpectedInteger, index~)
  }
  match repr {
    Some(text) => parse_decimal_int64(text, area, field, index)
    None => {
      if number.is_nan() ||
        number.is_inf() ||
        number != number.trunc() ||
        number < -9007199254740991.0 ||
        number > 9007199254740991.0 {
        raise InvalidValue(area~, field~, issue=ExpectedInteger, index~)
      }
      number.to_int64()
    }
  }
}

///|
fn parse_decimal_int64(
  text : StringView,
  area : DocumentArea,
  field : DocumentField,
  index : Int,
) -> Int64 raise ConfigDocumentError {
  if text.length() == 0 {
    raise InvalidValue(area~, field~, issue=ExpectedInteger, index~)
  }
  let negative = text[0] == '-'
  let start = if negative { 1 } else { 0 }
  if start == text.length() {
    raise InvalidValue(area~, field~, issue=ExpectedInteger, index~)
  }
  let limit = if negative { 0x8000000000000000UL } else { 0x7fffffffffffffffUL }
  let mut magnitude = 0UL
  for cursor in start..<text.length() {
    let unit = text[cursor]
    if unit < '0' || unit > '9' {
      raise InvalidValue(area~, field~, issue=ExpectedInteger, index~)
    }
    let digit = (unit.to_int() - '0').to_uint64()
    if magnitude > (limit - digit) / 10UL {
      raise InvalidValue(
        area~,
        field~,
        issue=RejectedByFocusedValidator,
        index~,
      )
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
