///|
pub(all) suberror AdapterError {
  ContractMismatch
  InvalidRange
  InvalidInstruction
  InvalidProgram
  InvalidState
  DeviceFault
  LaunchMismatch
}

///|
/// An admitted serialized contract, not a live device handle.
pub struct DeviceContract {
  priv capacity_bytes : UInt64
}

///|
pub fn parse_contract(text : String) -> DeviceContract raise AdapterError {
  if text.length() > 4096 {
    raise ContractMismatch
  }
  let value = @json.parse(text) catch { _ => raise ContractMismatch }
  let expected : Json = {
    "schema": "lunachip.pblaze3-device/v1",
    "target": "pblaze3-2gib",
    "control_abi": 65537,
    "device_id": 1346515760,
    "ddr_bytes": "2147483648",
    "descriptor_words": 16,
    "program_descriptors": 2048,
    "scratch_words": 32768,
    "vector_limit": 4096,
    "matrix_k_multiple": 64,
    "matrix_alignment_bytes": 128,
    "numeric_profile": "qwen-layer-ordered-fp32-approx-v1",
    "weight_layout": "bf16-row-major-or-packed16-v1",
  }
  if value != expected {
    raise ContractMismatch
  }
  { capacity_bytes: 2147483648UL, }
}

///|
pub fn DeviceContract::capacity_bytes(self : Self) -> UInt64 {
  self.capacity_bytes
}

///|
pub fn DeviceContract::check_range(
  self : Self,
  address : UInt64,
  bytes : UInt64,
) -> Unit raise AdapterError {
  if address > self.capacity_bytes || bytes > self.capacity_bytes - address {
    raise InvalidRange
  }
}
