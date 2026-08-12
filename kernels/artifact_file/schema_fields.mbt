///|
fn invalid_field(
  area : ManifestArea,
  field : ManifestField,
  issue : ManifestValueIssue,
  index? : Int = 0,
) -> KernelArtifactFileError {
  InvalidField(area~, field~, issue~, index~)
}

///|
fn validate_fields(
  object : Map[String, Json],
  area : ManifestArea,
  allowed : ArrayView[(String, ManifestField)],
) -> Unit raise KernelArtifactFileError {
  let mut index = 0
  for name, _ in object {
    let mut known = false
    for candidate in allowed {
      if name == candidate.0 {
        known = true
        break
      }
    }
    if !known {
      raise UnknownField(area~, index~)
    }
    index += 1
  }
  for entry in allowed {
    if !object.contains(entry.0) {
      raise MissingField(area~, field=entry.1)
    }
  }
}

///|
fn require_object(
  object : Map[String, Json],
  name : String,
  area : ManifestArea,
  field : ManifestField,
) -> Map[String, Json] raise KernelArtifactFileError {
  match object.get(name) {
    Some(Object(value)) => value
    Some(_) => raise invalid_field(area, field, ExpectedObject)
    None => raise MissingField(area~, field~)
  }
}

///|
fn require_array(
  object : Map[String, Json],
  name : String,
  area : ManifestArea,
  field : ManifestField,
) -> Array[Json] raise KernelArtifactFileError {
  match object.get(name) {
    Some(Array(value)) => value
    Some(_) => raise invalid_field(area, field, ExpectedArray)
    None => raise MissingField(area~, field~)
  }
}

///|
fn require_string(
  object : Map[String, Json],
  name : String,
  area : ManifestArea,
  field : ManifestField,
) -> String raise KernelArtifactFileError {
  match object.get(name) {
    Some(String(value)) => value
    Some(_) => raise invalid_field(area, field, ExpectedString)
    None => raise MissingField(area~, field~)
  }
}

///|
fn require_bool(
  object : Map[String, Json],
  name : String,
  area : ManifestArea,
  field : ManifestField,
) -> Bool raise KernelArtifactFileError {
  match object.get(name) {
    Some(True) => true
    Some(False) => false
    Some(_) => raise invalid_field(area, field, ExpectedBoolean)
    None => raise MissingField(area~, field~)
  }
}

///|
fn parse_nonnegative_int(
  value : Json,
  area : ManifestArea,
  field : ManifestField,
  index : Int,
) -> Int raise KernelArtifactFileError {
  guard value is Number(number, repr~) else {
    raise invalid_field(area, field, ExpectedInteger, index~)
  }
  match repr {
    Some(text) => {
      if text.is_empty() {
        raise invalid_field(area, field, ExpectedInteger, index~)
      }
      let mut result = 0
      for unit in text {
        if !(unit is ('0'..='9')) {
          raise invalid_field(area, field, ExpectedInteger, index~)
        }
        let digit = unit.to_int() - '0'
        if result > (2147483647 - digit) / 10 {
          raise invalid_field(area, field, ExpectedInteger, index~)
        }
        result = result * 10 + digit
      }
      result
    }
    None => {
      if number.is_nan() ||
        number.is_inf() ||
        number != number.trunc() ||
        number < 0.0 ||
        number > 2147483647.0 {
        raise invalid_field(area, field, ExpectedInteger, index~)
      }
      number.to_int()
    }
  }
}

///|
fn require_nonnegative_int(
  object : Map[String, Json],
  name : String,
  area : ManifestArea,
  field : ManifestField,
  index? : Int = 0,
) -> Int raise KernelArtifactFileError {
  match object.get(name) {
    Some(value) => parse_nonnegative_int(value, area, field, index)
    None => raise MissingField(area~, field~)
  }
}
