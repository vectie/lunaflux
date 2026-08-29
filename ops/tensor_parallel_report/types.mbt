///|
/// Typed operator projection constructed only from admitted topology and
/// capacity evidence.
pub struct TensorParallelCapacityReport {
  priv admission : @tensor_parallel_admission.TensorParallelRuntimeAdmission
} derive(Debug, Eq)
