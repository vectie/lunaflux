///|
const NUMERIC_SAFETENSORS_PREFIX_BYTES : Int = 8

///|
fn numeric_header_has_valid_envelope(header : BytesView) -> Bool {
  if header.is_empty() || header[0] != b'{' {
    return false
  }
  for index = header.length() - 1; index >= 0; index = index - 1 {
    match header[index] {
      b' ' => continue
      b'}' => return true
      _ => return false
    }
  }
  false
}

///|
fn decode_exact_int64(
  value : Json,
  ordinal : Int,
) -> Int64 raise NumericWeightFileError {
  guard value is Number(number, repr~) else { raise InvalidTensor(ordinal~) }
  match repr {
    Some(encoded) => {
      let integer : Int64 = @strconv.from_str(encoded) catch {
        _ => raise InvalidTensor(ordinal~)
      }
      integer
    }
    None => {
      if number.is_nan() {
        raise InvalidTensor(ordinal~)
      }
      let integer = number.to_int64()
      if integer.to_double() != number {
        raise InvalidTensor(ordinal~)
      }
      integer
    }
  }
}

///|
fn require_metadata_string(
  metadata : Map[String, Json],
  key : String,
) -> String raise NumericWeightFileError {
  guard metadata.get(key) is Some(String(value)) else {
    raise MissingDocumentMetadata
  }
  value
}

///|
fn validate_document_metadata(
  root : Map[String, Json],
  model_plan : @plan.ModelPlan,
) -> Unit raise NumericWeightFileError {
  guard root.get("__metadata__") is Some(Object(metadata)) else {
    raise MissingDocumentMetadata
  }
  if metadata.length() != 4 {
    raise UnexpectedDocumentMetadata
  }
  for key, value in metadata {
    if key != "format" &&
      key != "model_content_sha256" &&
      key != "model_plan_sha256" &&
      key != "numeric_schema_sha256" {
      raise UnexpectedDocumentMetadata
    }
    if !(value is String(_)) {
      raise MissingDocumentMetadata
    }
  }
  if require_metadata_string(metadata, "format") != NUMERIC_WEIGHT_FILE_FORMAT {
    raise FormatMismatch
  }
  if require_metadata_string(metadata, "model_content_sha256") !=
    model_plan.identity().content().as_hex() {
    raise ModelContentMismatch
  }
  if require_metadata_string(metadata, "model_plan_sha256") !=
    model_plan.identity().plan().as_hex() {
    raise ModelPlanMismatch
  }
  if require_metadata_string(metadata, "numeric_schema_sha256") !=
    model_plan.numeric_binding().digest().as_hex() {
    raise NumericSchemaMismatch
  }
}

///|
fn storage_dtype_name(
  storage : @numeric_contract.TensorStorageContract,
  ordinal : Int,
) -> String raise NumericWeightFileError {
  let dtype = storage.dtype()
  if dtype == @numeric_contract.StorageDType::bfloat16() {
    "BF16"
  } else if dtype == @numeric_contract.StorageDType::f8_e4m3_finite() {
    "F8_E4M3"
  } else if dtype == @numeric_contract.StorageDType::i8() {
    "I8"
  } else if dtype == @numeric_contract.StorageDType::f32() {
    "F32"
  } else {
    raise InvalidTensorDType(ordinal~)
  }
}

///|
fn require_tensor_field(
  tensor : Map[String, Json],
  name : String,
  ordinal : Int,
) -> Json raise NumericWeightFileError {
  match tensor.get(name) {
    Some(value) => value
    None => raise InvalidTensor(ordinal~)
  }
}

///|
fn validate_tensor_shape(
  value : Json,
  expected : @numeric_contract.NumericShape,
  ordinal : Int,
) -> Unit raise NumericWeightFileError {
  guard value is Array(dimensions) && dimensions.length() == expected.rank() else {
    raise InvalidTensorShape(ordinal~, axis=0)
  }
  for axis, dimension in dimensions {
    if decode_exact_int64(dimension, ordinal) !=
      expected.dimension(axis).unwrap() {
      raise InvalidTensorShape(ordinal~, axis~)
    }
  }
}

///|
fn validate_tensor_range(
  value : Json,
  expected_start : Int64,
  expected_length : Int64,
  payload_bytes : Int64,
  ordinal : Int,
) -> Int64 raise NumericWeightFileError {
  guard value is Array(offsets) && offsets.length() == 2 else {
    raise InvalidTensorRange(ordinal~)
  }
  let start = decode_exact_int64(offsets[0], ordinal)
  let end = decode_exact_int64(offsets[1], ordinal)
  if start != expected_start {
    raise PayloadNotContiguous(ordinal~)
  }
  if expected_length <= 0L ||
    start < 0L ||
    start > payload_bytes ||
    expected_length > payload_bytes - start {
    raise InvalidTensorRange(ordinal~)
  }
  if end != start + expected_length {
    raise InvalidTensorRange(ordinal~)
  }
  end
}

///|
fn parse_numeric_header(
  prefix_and_header : BytesView,
  payload_bytes : Int64,
  model_plan : @plan.ModelPlan,
  layout : @device_materialize.DeviceWeightLayout,
  limits : NumericWeightFileLimits,
) -> ReadOnlyArray[NumericTransferRegion] raise NumericWeightFileError {
  if prefix_and_header.length() <= NUMERIC_SAFETENSORS_PREFIX_BYTES {
    raise FileTruncated
  }
  let header = prefix_and_header[NUMERIC_SAFETENSORS_PREFIX_BYTES:]
  if !numeric_header_has_valid_envelope(header) {
    raise InvalidHeaderEnvelope
  }
  let text = @utf8.decode(header) catch { _ => raise InvalidHeaderEncoding }
  @json_guard.reject_duplicate_object_keys(text, max_depth=4) catch {
    _ => raise DuplicateHeaderKey
  }
  // The duplicate-key guard has already enforced depth four before the JSON
  // parser runs, so the parser cannot be driven into unbounded nesting here.
  let document = @json.parse(text) catch { _ => raise InvalidHeaderJson }
  guard document is Object(root) else { raise InvalidHeaderJson }
  validate_document_metadata(root, model_plan)
  let schema_tensors = model_plan.numeric_binding().schema().tensors()
  if schema_tensors.length() > limits.max_tensor_count() {
    raise InvalidLimit(TensorCount)
  }
  let actual_count = root.length() - 1
  if actual_count != schema_tensors.length() {
    raise TensorCountMismatch(
      expected=schema_tensors.length(),
      actual=actual_count,
    )
  }
  let regions : Array[NumericTransferRegion] = []
  let mut expected_start = 0L
  for ordinal, schema_tensor in schema_tensors {
    let name = "lunaflux.tensor.v1.\{ordinal}"
    guard root.get(name) is Some(Object(tensor)) else {
      raise MissingTensor(ordinal~)
    }
    if tensor.length() != 3 {
      raise UnexpectedTensorField(ordinal~)
    }
    for field, _ in tensor {
      if field != "dtype" && field != "shape" && field != "data_offsets" {
        raise UnexpectedTensorField(ordinal~)
      }
    }
    guard require_tensor_field(tensor, "dtype", ordinal) is String(dtype) else {
      raise InvalidTensorDType(ordinal~)
    }
    if dtype != storage_dtype_name(schema_tensor.storage(), ordinal) {
      raise InvalidTensorDType(ordinal~)
    }
    validate_tensor_shape(
      require_tensor_field(tensor, "shape", ordinal),
      schema_tensor.shape(),
      ordinal,
    )
    let byte_length = @numeric_contract.storage_byte_length(
      schema_tensor.storage().dtype(),
      schema_tensor.shape(),
    ) catch {
      _ => raise TensorSizeOverflow(ordinal~)
    }
    let end = validate_tensor_range(
      require_tensor_field(tensor, "data_offsets", ordinal),
      expected_start,
      byte_length,
      payload_bytes,
      ordinal,
    )
    let reference = @plan.TensorRef::new(ordinal) catch {
      _ => raise InvalidTensor(ordinal~)
    }
    guard layout.region(reference) is Some(destination) &&
      destination.byte_length() == byte_length else {
      raise InvalidTensorRange(ordinal~)
    }
    regions.push({
      ordinal,
      source_offset_bytes: prefix_and_header.length().to_int64() +
      expected_start,
      destination_offset_bytes: destination.offset_bytes(),
      byte_length,
    })
    expected_start = end
  }
  if expected_start != payload_bytes {
    raise TrailingPayload
  }
  // Count equality plus exact lookup of every canonical ordinal excludes all
  // aliases and unknown tensor names without an additional name table.
  ReadOnlyArray::from_array(regions)
}
