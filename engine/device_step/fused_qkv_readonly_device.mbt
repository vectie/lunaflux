///|
fn fused_qkv_readonly_live_device_join(
  expected_uuid~ : String,
  expected_pci~ : String,
  expected_name_hex~ : String,
  expected_memory_bytes~ : Int64,
  expected_compute_major~ : Int,
  expected_compute_minor~ : Int,
  expected_bf16~ : Bool,
  expected_cublas_lt~ : Bool,
  observed_uuid~ : String,
  observed_pci~ : String,
  observed_name_hex~ : String,
  observed_memory_bytes~ : Int64,
  observed_compute_major~ : Int,
  observed_compute_minor~ : Int,
  observed_bf16~ : Bool,
  observed_cublas_lt~ : Bool,
) -> FusedQkvReadOnlyLiveDeviceJoin raise FusedGroupedOperationError {
  let join : FusedQkvReadOnlyLiveDeviceJoin = {
    expected_uuid,
    expected_pci,
    expected_name_hex,
    expected_memory_bytes,
    expected_compute_major,
    expected_compute_minor,
    expected_bf16,
    expected_cublas_lt,
    observed_uuid,
    observed_pci,
    observed_name_hex,
    observed_memory_bytes,
    observed_compute_major,
    observed_compute_minor,
    observed_bf16,
    observed_cublas_lt,
  }
  if !join.is_exact() {
    raise MismatchedFusedGroupedOperation(RuntimeDevice)
  }
  join
}
