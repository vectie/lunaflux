///|
/// Builds an immutable report without probing hardware or acquiring runtime
/// authority.
pub fn build(
  admission : @tensor_parallel_admission.TensorParallelRuntimeAdmission,
) -> TensorParallelCapacityReport {
  { admission, }
}
