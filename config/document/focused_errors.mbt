///|
fn service_error_field(error : @service.ServiceConfigError) -> DocumentField {
  match error {
    Invalid(field~, issue=_) =>
      match field {
        ListenHost => ListenHostField
        ListenPort => ListenPortField
        MaxRequestBytes => MaxRequestBytesField
        GracefulDrainMilliseconds => GracefulDrainMillisecondsField
      }
  }
}

///|
fn model_error_field(error : @model.ModelConfigError) -> DocumentField {
  match error {
    Invalid(field~, issue=_, index=_) =>
      match field {
        ArtifactPath => ArtifactPathField
        ExpectedDigest => ExpectedDigestField
        ArchitectureAllowlist | ArchitectureName => ArchitectureAllowlistField
      }
  }
}

///|
fn model_error_index(error : @model.ModelConfigError) -> Int {
  match error {
    Invalid(field=_, issue=_, index~) => index
  }
}

///|
fn device_error_field(error : @device.DeviceConfigError) -> DocumentField {
  match error {
    Invalid(field~, issue=_, index=_) =>
      match field {
        VisibleDevices | DeviceId => VisibleDevicesField
        MemoryCeilingBytes => MemoryCeilingBytesField
      }
  }
}

///|
fn device_error_index(error : @device.DeviceConfigError) -> Int {
  match error {
    Invalid(field=_, issue=_, index~) => index
  }
}
