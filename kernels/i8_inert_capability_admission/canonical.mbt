///|
priv struct I8CanonicalWriter {
  output : Array[Byte]
  maximum : Int
  expected : Int
}

///|
fn I8CanonicalWriter::new(
  maximum : Int,
  expected : Int,
) -> I8CanonicalWriter raise I8InertKernelCapabilityAdmissionError {
  if expected < 0 || expected > maximum {
    raise I8InertKernelCapabilityRejected(CanonicalBytes)
  }
  { output: [], maximum, expected }
}

///|
fn I8CanonicalWriter::reserve(
  self : I8CanonicalWriter,
  additional : Int,
) -> Unit raise I8InertKernelCapabilityAdmissionError {
  let current = self.output.length()
  if additional < 0 ||
    current > self.maximum ||
    current > self.expected ||
    additional > self.maximum - current ||
    additional > self.expected - current {
    raise I8InertKernelCapabilityRejected(CanonicalBytes)
  }
}

///|
fn I8CanonicalWriter::append_byte(
  self : I8CanonicalWriter,
  value : Byte,
) -> Unit raise I8InertKernelCapabilityAdmissionError {
  self.reserve(1)
  self.output.push(value)
}

///|
fn I8CanonicalWriter::append_i64(
  self : I8CanonicalWriter,
  value : Int64,
) -> Unit raise I8InertKernelCapabilityAdmissionError {
  self.reserve(I8_ADMISSION_INTEGER_BYTES)
  for byte in value.reinterpret_as_uint64().to_le_bytes() {
    self.append_byte(byte)
  }
}

///|
fn I8CanonicalWriter::append_int(
  self : I8CanonicalWriter,
  value : Int,
) -> Unit raise I8InertKernelCapabilityAdmissionError {
  self.append_i64(Int64::from_int(value))
}

///|
fn i8_admission_is_lower_sha256(value : String) -> Bool {
  if value.length() != I8_ADMISSION_SHA256_HEX_BYTES {
    return false
  }
  for character in value {
    if !(character is ('0'..='9') || character is ('a'..='f')) {
      return false
    }
  }
  true
}

///|
fn I8CanonicalWriter::append_sha256_text(
  self : I8CanonicalWriter,
  value : String,
) -> Unit raise I8InertKernelCapabilityAdmissionError {
  if !i8_admission_is_lower_sha256(value) {
    raise I8InertKernelCapabilityRejected(CanonicalBytes)
  }
  let encoded = @utf8.encode(value)
  if encoded.length() != I8_ADMISSION_SHA256_HEX_BYTES {
    raise I8InertKernelCapabilityRejected(CanonicalBytes)
  }
  self.append_int(encoded.length())
  for byte in encoded {
    self.append_byte(byte)
  }
}

///|
fn I8CanonicalWriter::append_shape(
  self : I8CanonicalWriter,
  shape : @numeric_contract.NumericShape,
  expected_rank : Int,
) -> Unit raise I8InertKernelCapabilityAdmissionError {
  if shape.rank() != expected_rank {
    raise I8InertKernelCapabilityRejected(CanonicalBytes)
  }
  self.append_int(shape.rank())
  for dimension in shape.dimensions() {
    self.append_i64(dimension)
  }
  self.append_i64(shape.element_count())
}

///|
fn I8CanonicalWriter::append_entry_point(
  self : I8CanonicalWriter,
  entry_point : @catalog.AotKernelEntryPoint,
) -> Unit raise I8InertKernelCapabilityAdmissionError {
  self.append_sha256_text(entry_point.family().module_digest().sha256())
  self.append_int(entry_point.family().family_id().as_int())
  self.append_int(entry_point.entry_point_id().as_int())
}

///|
fn I8CanonicalWriter::finish(
  self : I8CanonicalWriter,
) -> Bytes raise I8InertKernelCapabilityAdmissionError {
  if self.output.length() != self.expected ||
    self.output.length() > self.maximum {
    raise I8InertKernelCapabilityRejected(CanonicalBytes)
  }
  Bytes::from_array(self.output)
}

///|
fn i8_operation_kind_identity(kind : @plan.OperationKind) -> Int {
  match kind {
    QkvProjection => 1
    OutputProjection => 2
    GatedMlp => 3
    LanguageModelHead => 4
    _ => 0
  }
}

///|
fn encode_i8_inert_kernel_capability_admission(
  model_identity : @spec.ModelIdentity,
  numeric_schema : @numeric_contract.ModelNumericSchemaDigest,
  device_numeric : @device.DeviceNumericCapabilityDigest,
  target : @catalog.DeviceTarget,
  catalog_version : @catalog.CatalogVersion,
  operations : ReadOnlyArray[I8InertOperationEvidence],
  maximum : Int,
  expected : Int,
) -> Bytes raise I8InertKernelCapabilityAdmissionError {
  let mut weight_count = 0
  for operation in operations {
    if operation.weights.length() > I8_ADMISSION_MAX_WEIGHTS - weight_count {
      raise I8InertKernelCapabilityRejected(CanonicalBytes)
    }
    weight_count += operation.weights.length()
  }
  guard i8_admission_v1_size_within(operations.length(), weight_count, maximum)
    is Some(recomputed) &&
    recomputed == expected else {
    raise I8InertKernelCapabilityRejected(CanonicalBytes)
  }
  let writer = I8CanonicalWriter::new(maximum, expected)
  for byte in b"lunaflux.i8-inert-kernel-capability-admission.v1\x00" {
    writer.append_byte(byte)
  }
  writer.append_int(1)
  writer.append_sha256_text(model_identity.content().as_hex())
  writer.append_sha256_text(model_identity.plan().as_hex())
  writer.append_sha256_text(numeric_schema.as_hex())
  writer.append_sha256_text(device_numeric.as_hex())
  writer.append_int(target.compute_major())
  writer.append_int(target.compute_minor())
  writer.append_int(if target.supports_bf16() { 1 } else { 0 })
  writer.append_int(if target.supports_cublas_lt() { 1 } else { 0 })
  writer.append_int(catalog_version.as_int())
  writer.append_int(operations.length())
  for operation in operations {
    writer.append_int(operation.operation.as_int())
    writer.append_int(i8_operation_kind_identity(operation.operation_kind))
    writer.append_int(operation.capability.as_int())
    writer.append_sha256_text(operation.execution.as_hex())
    writer.append_int(operation.semantic_version.as_int())
    writer.append_entry_point(operation.entry_point)
    writer.append_i64(operation.workspace_bytes)
    writer.append_i64(operation.workspace_alignment)
    writer.append_int(operation.weights.length())
    for weight in operation.weights {
      writer.append_int(weight.tensor.as_int())
      writer.append_sha256_text(weight.storage.as_hex())
      writer.append_shape(weight.shape, 2)
      writer.append_int(weight.scale.tensor().as_int())
      writer.append_sha256_text(weight.scale_storage.as_hex())
      writer.append_shape(weight.scale_shape, 1)
      writer.append_i64(weight.output_channel_count)
    }
  }
  writer.finish()
}
