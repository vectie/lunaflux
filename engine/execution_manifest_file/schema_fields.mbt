///|
fn invalid_field(
  area : ExecutionManifestArea,
  issue : ExecutionManifestIssue,
  index? : Int = 0,
) -> ExecutionManifestFileError {
  InvalidField(area~, issue~, index~)
}

///|
fn validate_fields(
  object : Map[String, Json],
  area : ExecutionManifestArea,
  allowed : ArrayView[String],
) -> Unit raise ExecutionManifestFileError {
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
  area : ExecutionManifestArea,
  index? : Int = 0,
) -> Map[String, Json] raise ExecutionManifestFileError {
  match object.get(name) {
    Some(Object(value)) => value
    Some(_) => raise invalid_field(area, ExpectedObject, index~)
    None => raise MissingField(area~, index~)
  }
}

///|
fn require_array(
  object : Map[String, Json],
  name : String,
  area : ExecutionManifestArea,
  index? : Int = 0,
) -> Array[Json] raise ExecutionManifestFileError {
  match object.get(name) {
    Some(Array(value)) => value
    Some(_) => raise invalid_field(area, ExpectedArray, index~)
    None => raise MissingField(area~, index~)
  }
}

///|
fn require_string(
  object : Map[String, Json],
  name : String,
  area : ExecutionManifestArea,
  index? : Int = 0,
) -> String raise ExecutionManifestFileError {
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
  area : ExecutionManifestArea,
  index? : Int = 0,
) -> Bool raise ExecutionManifestFileError {
  match object.get(name) {
    Some(True) => true
    Some(False) => false
    Some(_) => raise invalid_field(area, ExpectedBoolean, index~)
    None => raise MissingField(area~, index~)
  }
}

///|
fn parse_nonnegative_int64(
  value : Json,
  area : ExecutionManifestArea,
  index : Int,
) -> Int64 raise ExecutionManifestFileError {
  guard value is Number(number, repr~) else {
    raise invalid_field(area, ExpectedInteger, index~)
  }
  match repr {
    Some(text) => {
      if text.is_empty() {
        raise invalid_field(area, ExpectedInteger, index~)
      }
      let mut result = 0UL
      for unit in text {
        if !(unit is ('0'..='9')) {
          raise invalid_field(area, ExpectedInteger, index~)
        }
        let digit = (unit.to_int() - '0').to_uint64()
        if result > (0x7fffffffffffffffUL - digit) / 10UL {
          raise invalid_field(area, UnsupportedValue, index~)
        }
        result = result * 10UL + digit
      }
      result.reinterpret_as_int64()
    }
    None => {
      if number.is_nan() ||
        number.is_inf() ||
        number != number.trunc() ||
        number < 0.0 ||
        number > 9007199254740991.0 {
        raise invalid_field(area, ExpectedInteger, index~)
      }
      number.to_int64()
    }
  }
}

///|
fn require_nonnegative_int64(
  object : Map[String, Json],
  name : String,
  area : ExecutionManifestArea,
  index? : Int = 0,
) -> Int64 raise ExecutionManifestFileError {
  match object.get(name) {
    Some(value) => parse_nonnegative_int64(value, area, index)
    None => raise MissingField(area~, index~)
  }
}

///|
fn require_nonnegative_int(
  object : Map[String, Json],
  name : String,
  area : ExecutionManifestArea,
  index? : Int = 0,
) -> Int raise ExecutionManifestFileError {
  let value = require_nonnegative_int64(object, name, area, index~)
  if value > 2147483647L {
    raise invalid_field(area, UnsupportedValue, index~)
  }
  value.to_int()
}

///|
fn lexically_before(left : StringView, right : StringView) -> Bool {
  let common = left.length().min(right.length())
  for index in 0..<common {
    if left[index] < right[index] {
      return true
    }
    if left[index] > right[index] {
      return false
    }
  }
  left.length() < right.length()
}
