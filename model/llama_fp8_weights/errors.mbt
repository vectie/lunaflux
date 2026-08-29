///|
/// Bounded failures from dense-Llama FP8 numeric-weight inspection.
///
/// Plan construction and byte-level numeric-file authentication retain their
/// original typed causes without leaking an internal facade representation.
pub(all) suberror LlamaFp8WeightInspectionError {
  InvalidModelPlan(@plan.PlanValidationError)
  InvalidNumericWeightFile(@numeric_weight_file.NumericWeightFileError)
} derive(Debug, Eq)
