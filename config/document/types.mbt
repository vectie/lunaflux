///|
/// Stable configuration areas used by payload-safe document diagnostics.
pub(all) enum DocumentArea {
  RootArea
  ServiceArea
  ModelArea
  DeviceArea
} derive(Debug, Eq)

///|
/// Stable field vocabulary used by payload-safe document diagnostics.
pub(all) enum DocumentField {
  SchemaVersionField
  ServiceObjectField
  ListenHostField
  ListenPortField
  MaxRequestBytesField
  GracefulDrainMillisecondsField
  DiagnosticModeField
  ModelObjectField
  ArtifactPathField
  ExpectedDigestField
  ArchitectureAllowlistField
  DeviceObjectField
  VisibleDevicesField
  MemoryCeilingBytesField
} derive(Debug, Eq)

///|
/// Bounded reasons why a configuration value was rejected.
pub(all) enum DocumentValueIssue {
  ExpectedString
  ExpectedInteger
  ExpectedObject
  ExpectedArray
  InvalidArrayElement
  UnsupportedValue
  RejectedByFocusedValidator
} derive(Debug, Eq)

///|
/// Payload-safe failures from the bounded configuration document reader.
///
/// No variant carries source text, field names, paths, hostnames, or digests.
pub(all) suberror ConfigDocumentError {
  InvalidByteLimit
  ByteLimitExceeded(limit~ : Int)
  InvalidUtf8
  MalformedJson
  NonObjectRoot
  UnsupportedSchemaVersion
  NestingLimitExceeded(limit~ : Int)
  DuplicateField(depth~ : Int, index~ : Int)
  UnknownField(area~ : DocumentArea, index~ : Int)
  MissingField(area~ : DocumentArea, field~ : DocumentField)
  InvalidValue(
    area~ : DocumentArea,
    field~ : DocumentField,
    issue~ : DocumentValueIssue,
    index~ : Int
  )
} derive(Debug, Eq)

///|
/// Startup-only envelope containing focused immutable configuration records.
///
/// Runtime components should receive only the focused record they consume;
/// this document is not an engine-wide configuration object.
pub struct ConfigDocument {
  schema : @schema.DocumentSchema
  service : @service.ServiceConfig
  model : @model.ModelConfig
  device : @device.DeviceConfig
  resolved_plan : @resolved.ResolvedPlan
}

///|
pub fn ConfigDocument::schema(self : ConfigDocument) -> @schema.DocumentSchema {
  self.schema
}

///|
pub fn ConfigDocument::service(self : ConfigDocument) -> @service.ServiceConfig {
  self.service
}

///|
pub fn ConfigDocument::model(self : ConfigDocument) -> @model.ModelConfig {
  self.model
}

///|
pub fn ConfigDocument::device(self : ConfigDocument) -> @device.DeviceConfig {
  self.device
}

///|
pub fn ConfigDocument::resolved_plan(
  self : ConfigDocument,
) -> @resolved.ResolvedPlan {
  self.resolved_plan
}
