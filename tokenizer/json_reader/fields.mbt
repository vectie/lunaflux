///|
fn validate_fields(
  object : Map[String, Json],
  component : JsonComponent,
  allowed : ArrayView[String],
) -> Unit raise TokenizerJsonError {
  let mut field_index = 0
  for name, _ in object {
    if !allowed.any(candidate => candidate == name) {
      raise UnknownField(component~, index=field_index)
    }
    field_index += 1
  }
}

///|
fn require_field(
  object : Map[String, Json],
  name : String,
  component : JsonComponent,
  field : JsonField,
) -> Json raise TokenizerJsonError {
  match object.get(name) {
    Some(value) => value
    None => raise MissingField(component~, field~)
  }
}

///|
fn require_string(
  object : Map[String, Json],
  name : String,
  component : JsonComponent,
  field : JsonField,
  index : Int,
) -> String raise TokenizerJsonError {
  match require_field(object, name, component, field) {
    String(value) => value
    _ => raise InvalidField(component~, field~, index~)
  }
}

///|
fn require_bool(
  object : Map[String, Json],
  name : String,
  component : JsonComponent,
  field : JsonField,
  index : Int,
) -> Bool raise TokenizerJsonError {
  match require_field(object, name, component, field) {
    True => true
    False => false
    _ => raise InvalidField(component~, field~, index~)
  }
}

///|
fn require_exact_int(
  object : Map[String, Json],
  name : String,
  component : JsonComponent,
  field : JsonField,
  index : Int,
) -> Int raise TokenizerJsonError {
  guard require_field(object, name, component, field) is Number(number, repr~) else {
    raise InvalidField(component~, field~, index~)
  }
  match repr {
    Some(encoded) => {
      let integer : Int = @strconv.from_str(encoded) catch {
        _ => raise InvalidField(component~, field~, index~)
      }
      integer
    }
    None => {
      if number.is_nan() {
        raise InvalidField(component~, field~, index~)
      }
      let integer = number.to_int()
      if integer.to_double() != number {
        raise InvalidField(component~, field~, index~)
      }
      integer
    }
  }
}

///|
fn require_null(
  object : Map[String, Json],
  name : String,
  component : JsonComponent,
  field : JsonField,
  unsupported : UnsupportedFeature,
) -> Unit raise TokenizerJsonError {
  match require_field(object, name, component, field) {
    Null => ()
    _ => raise UnsupportedSemantics(component~, feature=unsupported)
  }
}
