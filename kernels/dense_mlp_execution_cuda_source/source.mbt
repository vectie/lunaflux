///|
pub struct DenseMlpCudaSource {
  priv plan : @precision.DenseSwiGluPrecision
  priv prefix : String
  priv source : Bytes
}

///|
pub fn dense_mlp_source(
  plan : @precision.DenseSwiGluPrecision,
  symbol_prefix~ : String,
) -> DenseMlpCudaSource raise {
  let (hidden, intermediate, rows) = plan.geometry()
  {
    plan,
    prefix: symbol_prefix,
    source: @lowering.staged_separate_swiglu_bf16_source(
      hidden_size=hidden,
      intermediate_size=intermediate,
      maximum_rows=rows,
      function_prefix=symbol_prefix,
    ),
  }
}

///|
pub fn DenseMlpCudaSource::plan(self : Self) -> @precision.DenseSwiGluPrecision {
  self.plan
}

///|
pub fn DenseMlpCudaSource::source_bytes(self : Self) -> Bytes {
  self.source
}

///|
pub fn DenseMlpCudaSource::symbol(
  self : Self,
  stage : Int,
) -> String raise @precision.PrecisionPlanError {
  let suffix = match stage {
    0 => "_gate_up"
    1 => "_down"
    _ => raise @precision.PrecisionPlanError::IllegalPlacement
  }
  self.prefix + suffix
}
