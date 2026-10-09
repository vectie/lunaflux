///|
pub fn CudaOfflineCompileEvidenceLimits::new(
  maximum_module_bytes~ : Int64,
  maximum_receipt_bytes~ : Int,
  maximum_symbol_bytes~ : Int,
) -> Self raise CudaOfflineCompileEvidenceError {
  if maximum_module_bytes <= 0L ||
    maximum_module_bytes > 2147483647L ||
    maximum_receipt_bytes <= 0 ||
    maximum_receipt_bytes > 1048576 ||
    maximum_symbol_bytes <= 0 ||
    maximum_symbol_bytes > 1024 {
    raise InvalidLimit
  }
  { maximum_module_bytes, maximum_receipt_bytes, maximum_symbol_bytes, }
}
