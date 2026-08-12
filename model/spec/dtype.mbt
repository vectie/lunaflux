///|
/// Numeric storage types recognized by the dense-model metadata validator.
///
/// Recognition is intentionally wider than v1 execution support so unsupported
/// precision is reported explicitly during model validation.
pub(all) enum ModelDType {
  BFloat16
  Float16
  Float32
} derive(Debug, Eq)

///|
/// Parses a canonical model dtype name without accepting aliases or fallbacks.
pub fn ModelDType::parse(name : StringView) -> ModelDType raise ModelSpecError {
  match name {
    "BF16" => BFloat16
    "F16" => Float16
    "F32" => Float32
    _ => raise UnknownDType
  }
}
