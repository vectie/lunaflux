///|
/// Configuration document versions understood by this runtime.
pub(all) enum SchemaVersion {
  V1Alpha1
} derive(Debug, Eq)

///|
/// Required top-level sections in a LunaFlux configuration document.
pub(all) enum Section {
  SchemaVersion
  Service
  Model
  Device
} derive(Debug, Eq)

///|
/// Bounded failures produced while admitting a configuration document.
///
/// Field content is deliberately excluded so diagnostics cannot echo an
/// attacker-controlled name. `UnknownField` and `DuplicateField` identify the
/// position in the bounded top-level field list.
pub(all) suberror SchemaError {
  TooManyFields(limit~ : Int)
  UnsupportedVersion
  UnknownField(index~ : Int)
  DuplicateField(index~ : Int, first_index~ : Int)
  MissingSection(Section)
} derive(Debug, Eq)

///|
/// Validated schema envelope for one configuration document.
pub struct DocumentSchema {
  version : SchemaVersion
}

///|
/// Canonical textual schema version.
pub fn SchemaVersion::name(self : SchemaVersion) -> String {
  match self {
    V1Alpha1 => "lunaflux.config.v1alpha1"
  }
}

///|
/// Returns the admitted document version.
pub fn DocumentSchema::version(self : DocumentSchema) -> SchemaVersion {
  self.version
}

///|
/// Validates the schema version and complete top-level field vocabulary.
///
/// This is the dependency-free boundary used by a future bounded JSON reader:
/// it rejects unknown, duplicate, and missing fields before any focused
/// configuration record is constructed.
pub fn validate_document(
  schema_version : StringView,
  top_level_fields : ArrayView[String],
) -> DocumentSchema raise SchemaError {
  let limit = 4
  if top_level_fields.length() > limit {
    raise TooManyFields(limit~)
  }
  if schema_version != "lunaflux.config.v1alpha1" {
    raise UnsupportedVersion
  }
  let mut seen_version = -1
  let mut seen_service = -1
  let mut seen_model = -1
  let mut seen_device = -1
  for index, field in top_level_fields {
    let previous = match field {
      "schema_version" => seen_version
      "service" => seen_service
      "model" => seen_model
      "device" => seen_device
      _ => raise UnknownField(index~)
    }
    if previous >= 0 {
      raise DuplicateField(index~, first_index=previous)
    }
    match field {
      "schema_version" => seen_version = index
      "service" => seen_service = index
      "model" => seen_model = index
      "device" => seen_device = index
      _ => ()
    }
  }
  if seen_version < 0 {
    raise MissingSection(SchemaVersion)
  }
  if seen_service < 0 {
    raise MissingSection(Service)
  }
  if seen_model < 0 {
    raise MissingSection(Model)
  }
  if seen_device < 0 {
    raise MissingSection(Device)
  }
  { version: V1Alpha1 }
}
