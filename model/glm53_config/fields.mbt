///|
fn invalid(field : Glm53ConfigField) -> Glm53ConfigError {
  InvalidValue(field)
}

///|
fn require_value(
  object : Map[String, Json],
  name : String,
  field : Glm53ConfigField,
) -> Json raise Glm53ConfigError {
  match object.get(name) {
    Some(value) => value
    None => raise MissingField(field)
  }
}

///|
fn require_object(
  object : Map[String, Json],
  name : String,
  field : Glm53ConfigField,
) -> Map[String, Json] raise Glm53ConfigError {
  match require_value(object, name, field) {
    Object(value) => value
    _ => raise invalid(field)
  }
}

///|
fn require_string(
  object : Map[String, Json],
  name : String,
  field : Glm53ConfigField,
) -> String raise Glm53ConfigError {
  match require_value(object, name, field) {
    String(value) => value
    _ => raise invalid(field)
  }
}

///|
fn require_int(
  object : Map[String, Json],
  name : String,
  field : Glm53ConfigField,
) -> Int raise Glm53ConfigError {
  match require_value(object, name, field) {
    Number(value, repr=_) if !value.is_nan() &&
      !value.is_inf() &&
      value == value.trunc() &&
      value >= -2147483648.0 &&
      value <= 2147483647.0 => value.to_int()
    _ => raise invalid(field)
  }
}

///|
fn require_number(
  object : Map[String, Json],
  name : String,
  field : Glm53ConfigField,
) -> Double raise Glm53ConfigError {
  match require_value(object, name, field) {
    Number(value, repr=_) if !value.is_nan() && !value.is_inf() => value
    _ => raise invalid(field)
  }
}

///|
fn require_bool(
  object : Map[String, Json],
  name : String,
  field : Glm53ConfigField,
) -> Bool raise Glm53ConfigError {
  match require_value(object, name, field) {
    True => true
    False => false
    _ => raise invalid(field)
  }
}

///|
fn check_int(
  object : Map[String, Json],
  name : String,
  expected : Int,
  field : Glm53ConfigField,
) -> Unit raise Glm53ConfigError {
  if require_int(object, name, field) != expected {
    raise invalid(field)
  }
}

///|
fn check_number(
  object : Map[String, Json],
  name : String,
  expected : Double,
  field : Glm53ConfigField,
) -> Unit raise Glm53ConfigError {
  if require_number(object, name, field) != expected {
    raise invalid(field)
  }
}

///|
fn check_string(
  object : Map[String, Json],
  name : String,
  expected : String,
  field : Glm53ConfigField,
) -> Unit raise Glm53ConfigError {
  if require_string(object, name, field) != expected {
    raise invalid(field)
  }
}

///|
fn check_bool(
  object : Map[String, Json],
  name : String,
  expected : Bool,
  field : Glm53ConfigField,
) -> Unit raise Glm53ConfigError {
  if require_bool(object, name, field) != expected {
    raise invalid(field)
  }
}

///|
fn check_single_architecture(
  object : Map[String, Json],
  expected : String,
) -> Unit raise Glm53ConfigError {
  match require_value(object, "architectures", Architecture) {
    Array([String(actual)]) if actual == expected => ()
    _ => raise invalid(Architecture)
  }
}

///|
fn require_string_array(
  object : Map[String, Json],
  name : String,
  field : Glm53ConfigField,
) -> Array[String] raise Glm53ConfigError {
  guard require_value(object, name, field) is Array(values) else {
    raise invalid(field)
  }
  let output = []
  for value in values {
    match value {
      String(item) => output.push(item)
      _ => raise invalid(field)
    }
  }
  output
}

///|
fn require_int_array(
  object : Map[String, Json],
  name : String,
  field : Glm53ConfigField,
) -> Array[Int] raise Glm53ConfigError {
  guard require_value(object, name, field) is Array(values) else {
    raise invalid(field)
  }
  let output = []
  for value in values {
    match value {
      Number(number, repr=_) if number == number.trunc() &&
        number >= -2147483648.0 &&
        number <= 2147483647.0 => output.push(number.to_int())
      _ => raise invalid(field)
    }
  }
  output
}

///|
fn check_string_schedule(
  object : Map[String, Json],
  name : String,
  length : Int,
  expected_at : (Int) -> String,
  field : Glm53ConfigField,
) -> Unit raise Glm53ConfigError {
  let values = require_string_array(object, name, field)
  if values.length() != length {
    raise invalid(field)
  }
  for index in 0..<length {
    if values[index] != expected_at(index) {
      raise invalid(field)
    }
  }
}

///|
fn check_int_schedule(
  object : Map[String, Json],
  name : String,
  length : Int,
  expected_at : (Int) -> Int,
  field : Glm53ConfigField,
) -> Unit raise Glm53ConfigError {
  let values = require_int_array(object, name, field)
  if values.length() != length {
    raise invalid(field)
  }
  for index in 0..<length {
    if values[index] != expected_at(index) {
      raise invalid(field)
    }
  }
}

///|
fn check_int_array(
  object : Map[String, Json],
  name : String,
  expected : ArrayView[Int],
  field : Glm53ConfigField,
) -> Unit raise Glm53ConfigError {
  let values = require_int_array(object, name, field)
  if values.length() != expected.length() {
    raise invalid(field)
  }
  for index, value in expected {
    if values[index] != value {
      raise invalid(field)
    }
  }
}

///|
fn check_null(
  object : Map[String, Json],
  name : String,
  field : Glm53ConfigField,
) -> Unit raise Glm53ConfigError {
  match require_value(object, name, field) {
    Null => ()
    _ => raise invalid(field)
  }
}
