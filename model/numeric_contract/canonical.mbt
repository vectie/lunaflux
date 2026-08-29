///|
fn tensor_storage_domain() -> Bytes {
  b"lunaflux.tensor-storage-contract.v1\x00"
}

///|
fn operation_execution_domain() -> Bytes {
  b"lunaflux.operation-execution-contract.v2\x00"
}

///|
const TENSOR_STORAGE_CANONICAL_LENGTH : Int = 45

///|
const OPERATION_EXECUTION_CANONICAL_LENGTH : Int = 51

///|
fn write_tensor_storage_contract(
  writer : NumericCanonicalWriter,
  contract : TensorStorageContract,
) -> Unit raise NumericContractError {
  writer.write_bytes(tensor_storage_domain())
  writer.write_byte(contract.dtype.value.to_byte())
  writer.write_byte(contract.layout.value.to_byte())
  writer.write_byte(contract.encoding.value.to_byte())
  writer.write_byte(contract.scale_granularity.value.to_byte())
  writer.write_byte(contract.zero_point.value.to_byte())
  writer.write_byte(contract.codebook_layout.value.to_byte())
  writer.write_byte(contract.conversion.rounding.value.to_byte())
  writer.write_byte(contract.conversion.saturation.value.to_byte())
  writer.write_byte(contract.conversion.finite_values.value.to_byte())
}

///|
fn tensor_storage_contract_byte_at(
  contract : TensorStorageContract,
  index : Int,
) -> Byte {
  let domain = tensor_storage_domain()
  if index < domain.length() {
    return domain[index]
  }
  match index - domain.length() {
    0 => contract.dtype.value.to_byte()
    1 => contract.layout.value.to_byte()
    2 => contract.encoding.value.to_byte()
    3 => contract.scale_granularity.value.to_byte()
    4 => contract.zero_point.value.to_byte()
    5 => contract.codebook_layout.value.to_byte()
    6 => contract.conversion.rounding.value.to_byte()
    7 => contract.conversion.saturation.value.to_byte()
    8 => contract.conversion.finite_values.value.to_byte()
    _ => abort("tensor storage canonical byte index out of bounds")
  }
}

///|
fn encode_tensor_storage_contract(contract : TensorStorageContract) -> Bytes {
  materialize_numeric_canonical(contract.canonical_length, index => {
    tensor_storage_contract_byte_at(contract, index)
  })
}

///|
fn write_operation_execution_contract(
  writer : NumericCanonicalWriter,
  contract : OperationExecutionContract,
) -> Unit raise NumericContractError {
  writer.write_bytes(operation_execution_domain())
  writer.write_byte(contract.activation_input.value.to_byte())
  writer.write_byte(contract.activation_compute.value.to_byte())
  writer.write_byte(contract.activation_scale.value.to_byte())
  writer.write_byte(contract.tensor_input.value.to_byte())
  writer.write_byte(contract.accumulator.value.to_byte())
  writer.write_byte(contract.output.value.to_byte())
  writer.write_byte(contract.conversion.rounding.value.to_byte())
  writer.write_byte(contract.conversion.saturation.value.to_byte())
  writer.write_byte(contract.conversion.finite_values.value.to_byte())
  writer.write_byte(contract.order.value.to_byte())
}

///|
fn operation_execution_contract_byte_at(
  contract : OperationExecutionContract,
  index : Int,
) -> Byte {
  let domain = operation_execution_domain()
  if index < domain.length() {
    return domain[index]
  }
  match index - domain.length() {
    0 => contract.activation_input.value.to_byte()
    1 => contract.activation_compute.value.to_byte()
    2 => contract.activation_scale.value.to_byte()
    3 => contract.tensor_input.value.to_byte()
    4 => contract.accumulator.value.to_byte()
    5 => contract.output.value.to_byte()
    6 => contract.conversion.rounding.value.to_byte()
    7 => contract.conversion.saturation.value.to_byte()
    8 => contract.conversion.finite_values.value.to_byte()
    9 => contract.order.value.to_byte()
    _ => abort("operation execution canonical byte index out of bounds")
  }
}

///|
fn encode_operation_execution_contract(
  contract : OperationExecutionContract,
) -> Bytes {
  materialize_numeric_canonical(contract.canonical_length, index => {
    operation_execution_contract_byte_at(contract, index)
  })
}

///|
fn build_tensor_storage_contract(
  dtype : StorageDType,
  layout : StorageLayout,
  encoding : StorageEncoding,
  scale_granularity : ScaleGranularity,
  zero_point : ZeroPointMode,
  codebook_layout : CodebookLayout,
  conversion : NumericConversion,
) -> TensorStorageContract {
  let placeholder = {
    dtype,
    layout,
    encoding,
    scale_granularity,
    zero_point,
    codebook_layout,
    conversion,
    canonical_length: TENSOR_STORAGE_CANONICAL_LENGTH,
    digest: { value: "" },
  }
  let writer = NumericCanonicalWriter::new_hashing(
    TENSOR_STORAGE_CANONICAL_LENGTH,
  )
  try! write_tensor_storage_contract(writer, placeholder)
  { ..placeholder, digest: { value: try! writer.finish_hex() } }
}

///|
fn build_operation_execution_contract(
  activation_input : ComputeDType,
  activation_compute : ComputeDType,
  activation_scale : ActivationScalePolicy,
  tensor_input : ComputeDType,
  accumulator : AccumulatorDType,
  output : ComputeDType,
  conversion : NumericConversion,
  order : OperationOrderPolicy,
) -> OperationExecutionContract {
  let placeholder = {
    activation_input,
    activation_compute,
    activation_scale,
    tensor_input,
    accumulator,
    output,
    conversion,
    order,
    canonical_length: OPERATION_EXECUTION_CANONICAL_LENGTH,
    digest: { value: "" },
  }
  let writer = NumericCanonicalWriter::new_hashing(
    OPERATION_EXECUTION_CANONICAL_LENGTH,
  )
  try! write_operation_execution_contract(writer, placeholder)
  { ..placeholder, digest: { value: try! writer.finish_hex() } }
}

///|
fn is_lower_sha256(value : StringView) -> Bool {
  if value.length() != 64 {
    return false
  }
  for unit in value {
    if !(unit is ('0'..='9' | 'a'..='f')) {
      return false
    }
  }
  true
}

///|
pub fn TensorStorageDigest::from_sha256(
  value : String,
) -> TensorStorageDigest raise NumericContractError {
  if !is_lower_sha256(value) {
    raise InvalidNumericContract(Digest)
  }
  { value, }
}

///|
pub fn TensorStorageDigest::as_hex(self : TensorStorageDigest) -> String {
  self.value
}

///|
pub fn OperationExecutionDigest::from_sha256(
  value : String,
) -> OperationExecutionDigest raise NumericContractError {
  if !is_lower_sha256(value) {
    raise InvalidNumericContract(Digest)
  }
  { value, }
}

///|
pub fn OperationExecutionDigest::as_hex(
  self : OperationExecutionDigest,
) -> String {
  self.value
}

///|
fn digest_bytes(bytes : Bytes) -> String {
  let hasher = @canonical_sha256.CanonicalSha256::new()
  if !hasher.update_bytes(bytes) {
    abort("fresh canonical hasher rejected input")
  }
  hasher.finalize_hex()
}

///|
fn require_domain(
  canonical : Bytes,
  domain : Bytes,
  tag_count : Int,
) -> Int raise NumericContractError {
  if canonical.length() != domain.length() + tag_count {
    raise InvalidNumericContract(CanonicalBytes)
  }
  for index in 0..<domain.length() {
    if canonical[index] != domain[index] {
      raise InvalidNumericContract(CanonicalBytes)
    }
  }
  domain.length()
}

///|
fn decoded_conversion(
  canonical : Bytes,
  offset : Int,
) -> NumericConversion raise NumericContractError {
  NumericConversion::new(
    rounding={ value: canonical[offset].to_int() },
    saturation={ value: canonical[offset + 1].to_int() },
    finite_values={ value: canonical[offset + 2].to_int() },
  )
}

///|
pub fn load_tensor_storage_contract(
  canonical : Bytes,
  expected_digest : TensorStorageDigest,
) -> TensorStorageContract raise NumericContractError {
  if digest_bytes(canonical) != expected_digest.value {
    raise TensorStorageDigestMismatch
  }
  let offset = require_domain(canonical, tensor_storage_domain(), 9)
  let contract = TensorStorageContract::new(
    dtype={ value: canonical[offset].to_int() },
    layout={ value: canonical[offset + 1].to_int() },
    encoding={ value: canonical[offset + 2].to_int() },
    scale_granularity={ value: canonical[offset + 3].to_int() },
    zero_point={ value: canonical[offset + 4].to_int() },
    codebook_layout={ value: canonical[offset + 5].to_int() },
    conversion=decoded_conversion(canonical, offset + 6),
  )
  if contract.canonical_bytes() != canonical {
    raise InvalidNumericContract(CanonicalBytes)
  }
  contract
}

///|
pub fn load_operation_execution_contract(
  canonical : Bytes,
  expected_digest : OperationExecutionDigest,
) -> OperationExecutionContract raise NumericContractError {
  if digest_bytes(canonical) != expected_digest.value {
    raise OperationExecutionDigestMismatch
  }
  let offset = require_domain(canonical, operation_execution_domain(), 10)
  let contract = OperationExecutionContract::new(
    activation_input={ value: canonical[offset].to_int() },
    activation_compute={ value: canonical[offset + 1].to_int() },
    activation_scale={ value: canonical[offset + 2].to_int() },
    tensor_input={ value: canonical[offset + 3].to_int() },
    accumulator={ value: canonical[offset + 4].to_int() },
    output={ value: canonical[offset + 5].to_int() },
    conversion=decoded_conversion(canonical, offset + 6),
    order={ value: canonical[offset + 9].to_int() },
  )
  if contract.canonical_bytes() != canonical {
    raise InvalidNumericContract(CanonicalBytes)
  }
  contract
}
