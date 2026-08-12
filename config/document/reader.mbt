///|
const JSON_NESTING_LIMIT : Int = 32

///|
/// Parses a UTF-8 configuration document after enforcing `max_bytes` before
/// decoding or JSON allocation.
pub fn parse_bytes(
  document : BytesView,
  max_bytes~ : Int,
) -> ConfigDocument raise ConfigDocumentError {
  validate_limit(max_bytes)
  if document.length() > max_bytes {
    raise ByteLimitExceeded(limit=max_bytes)
  }
  let text = @utf8.decode(document) catch { _ => raise InvalidUtf8 }
  parse_text(text)
}

///|
/// Parses a configuration string after validating its UTF-16 structure and
/// enforcing the encoded UTF-8 byte bound without first allocating an encoded
/// copy.
pub fn parse_string(
  document : StringView,
  max_bytes~ : Int,
) -> ConfigDocument raise ConfigDocumentError {
  validate_limit(max_bytes)
  validate_utf8_size(document, max_bytes)
  parse_text(document)
}

///|
fn validate_limit(max_bytes : Int) -> Unit raise ConfigDocumentError {
  if max_bytes <= 0 {
    raise InvalidByteLimit
  }
}

///|
fn validate_utf8_size(
  input : StringView,
  max_bytes : Int,
) -> Unit raise ConfigDocumentError {
  let mut index = 0
  let mut byte_count = 0
  while index < input.length() {
    let unit = input[index]
    let (width, next_index) = if unit <= 0x7f {
      (1, index + 1)
    } else if unit <= 0x7ff {
      (2, index + 1)
    } else if unit.is_leading_surrogate() {
      if index + 1 >= input.length() ||
        !input[index + 1].is_trailing_surrogate() {
        raise InvalidUtf8
      }
      (4, index + 2)
    } else if unit.is_trailing_surrogate() {
      raise InvalidUtf8
    } else {
      (3, index + 1)
    }
    if byte_count > max_bytes - width {
      raise ByteLimitExceeded(limit=max_bytes)
    }
    byte_count += width
    index = next_index
  }
}

///|
fn parse_text(
  document : StringView,
) -> ConfigDocument raise ConfigDocumentError {
  @json_guard.reject_duplicate_object_keys(
    document,
    max_depth=JSON_NESTING_LIMIT,
  ) catch {
    InvalidDepthLimit => raise MalformedJson
    DepthLimitExceeded(limit~) => raise NestingLimitExceeded(limit~)
    DuplicateObjectKey(depth~, index~) => raise DuplicateField(depth~, index~)
    MalformedStructure => raise MalformedJson
  }
  let parsed = @json.parse(document) catch { _ => raise MalformedJson }
  guard parsed is Object(root) else { raise NonObjectRoot }
  let schema = parse_schema(root)
  let service = parse_service(root["service"])
  let model = parse_model(root["model"])
  let device = parse_device(root["device"])
  let resolved_plan = @resolved.resolve(service, model, device)
  { schema, service, model, device, resolved_plan }
}

///|
fn parse_schema(
  root : Map[String, Json],
) -> @schema.DocumentSchema raise ConfigDocumentError {
  let version = match root.get("schema_version") {
    None => raise MissingField(area=RootArea, field=SchemaVersionField)
    Some(String(value)) => value
    Some(_) =>
      raise InvalidValue(
        area=RootArea,
        field=SchemaVersionField,
        issue=ExpectedString,
        index=0,
      )
  }
  let fields : Array[String] = []
  for field, _ in root {
    fields.push(field)
  }
  @schema.validate_document(version, fields) catch {
    TooManyFields(limit~) => raise UnknownField(area=RootArea, index=limit)
    UnsupportedVersion => raise UnsupportedSchemaVersion
    UnknownField(index~) => raise UnknownField(area=RootArea, index~)
    DuplicateField(index~, first_index=_) =>
      raise UnknownField(area=RootArea, index~)
    MissingSection(section) =>
      raise MissingField(area=RootArea, field=section_field(section))
  }
}

///|
fn section_field(section : @schema.Section) -> DocumentField {
  match section {
    SchemaVersion => SchemaVersionField
    Service => ServiceObjectField
    Model => ModelObjectField
    Device => DeviceObjectField
  }
}

///|
fn parse_service(
  value : Json,
) -> @service.ServiceConfig raise ConfigDocumentError {
  guard value is Object(object) else {
    raise InvalidValue(
      area=RootArea,
      field=ServiceObjectField,
      issue=ExpectedObject,
      index=0,
    )
  }
  let allowed : Array[(String, DocumentField)] = [
    ("listen_host", ListenHostField),
    ("listen_port", ListenPortField),
    ("max_request_bytes", MaxRequestBytesField),
    ("graceful_drain_milliseconds", GracefulDrainMillisecondsField),
    ("diagnostic_mode", DiagnosticModeField),
  ]
  validate_fields(object, ServiceArea, allowed)
  let listen_host = require_string(
    object,
    ServiceArea,
    "listen_host",
    ListenHostField,
  )
  let listen_port = require_int(
    object,
    ServiceArea,
    "listen_port",
    ListenPortField,
  )
  let max_request_bytes = require_int(
    object,
    ServiceArea,
    "max_request_bytes",
    MaxRequestBytesField,
  )
  let graceful_drain_milliseconds = require_int(
    object,
    ServiceArea,
    "graceful_drain_milliseconds",
    GracefulDrainMillisecondsField,
  )
  let diagnostic_text = require_string(
    object,
    ServiceArea,
    "diagnostic_mode",
    DiagnosticModeField,
  )
  let diagnostic_mode : @service.DiagnosticMode = match diagnostic_text {
    "normal" => Normal
    "verbose" => Verbose
    _ =>
      raise InvalidValue(
        area=ServiceArea,
        field=DiagnosticModeField,
        issue=UnsupportedValue,
        index=0,
      )
  }
  @service.new(
    listen_host, listen_port, max_request_bytes, graceful_drain_milliseconds, diagnostic_mode,
  ) catch {
    error =>
      raise InvalidValue(
        area=ServiceArea,
        field=service_error_field(error),
        issue=RejectedByFocusedValidator,
        index=0,
      )
  }
}

///|
fn parse_model(value : Json) -> @model.ModelConfig raise ConfigDocumentError {
  guard value is Object(object) else {
    raise InvalidValue(
      area=RootArea,
      field=ModelObjectField,
      issue=ExpectedObject,
      index=0,
    )
  }
  let allowed : Array[(String, DocumentField)] = [
    ("artifact_path", ArtifactPathField),
    ("expected_digest", ExpectedDigestField),
    ("architecture_allowlist", ArchitectureAllowlistField),
  ]
  validate_fields(object, ModelArea, allowed)
  let artifact_path = require_string(
    object,
    ModelArea,
    "artifact_path",
    ArtifactPathField,
  )
  let expected_digest = require_string(
    object,
    ModelArea,
    "expected_digest",
    ExpectedDigestField,
  )
  let architecture_allowlist = require_string_array(
    object,
    ModelArea,
    "architecture_allowlist",
    ArchitectureAllowlistField,
  )
  @model.new(artifact_path, expected_digest, architecture_allowlist) catch {
    error =>
      raise InvalidValue(
        area=ModelArea,
        field=model_error_field(error),
        issue=RejectedByFocusedValidator,
        index=model_error_index(error),
      )
  }
}

///|
fn parse_device(value : Json) -> @device.DeviceConfig raise ConfigDocumentError {
  guard value is Object(object) else {
    raise InvalidValue(
      area=RootArea,
      field=DeviceObjectField,
      issue=ExpectedObject,
      index=0,
    )
  }
  let allowed : Array[(String, DocumentField)] = [
    ("visible_devices", VisibleDevicesField),
    ("memory_ceiling_bytes", MemoryCeilingBytesField),
  ]
  validate_fields(object, DeviceArea, allowed)
  let visible_devices = require_int_array(
    object,
    DeviceArea,
    "visible_devices",
    VisibleDevicesField,
  )
  let memory_ceiling_bytes = require_int64(
    object,
    DeviceArea,
    "memory_ceiling_bytes",
    MemoryCeilingBytesField,
  )
  @device.new(visible_devices, memory_ceiling_bytes) catch {
    error =>
      raise InvalidValue(
        area=DeviceArea,
        field=device_error_field(error),
        issue=RejectedByFocusedValidator,
        index=device_error_index(error),
      )
  }
}
