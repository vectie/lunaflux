///|
pub struct DenseMlpCudaSource {
  priv plan : @precision.DenseSwiGluPrecision
  priv prefix : String
  priv source : Bytes
  priv order : ReadOnlyArray[Int]
}

///|
pub fn dense_mlp_source(
  plan : @precision.DenseSwiGluPrecision,
  symbol_prefix~ : String,
) -> DenseMlpCudaSource raise {
  let (hidden, intermediate, rows) = plan.geometry()
  let program = plan.program()
  let order = program
    .lower(Ordered)
    .commands(ComputeLane)
    .map(command => {
      match command {
        Execute(step~) =>
          match program.steps()[step].action() {
            Compute(stage) => stage
            _ => raise @precision.PrecisionPlanError::NoImplementation
          }
        _ => raise @precision.PrecisionPlanError::NoImplementation
      }
    })
  {
    plan,
    order,
    prefix: symbol_prefix,
    source: @lowering.staged_separate_swiglu_bf16_source(
      hidden_size=hidden,
      intermediate_size=intermediate,
      maximum_rows=rows,
      function_prefix=symbol_prefix,
      publication=plan.publication(),
    ),
  }
}

///|
pub fn DenseMlpCudaSource::execution_order(self : Self) -> ReadOnlyArray[Int] {
  self.order
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
    2 if self.plan.publication() == ReduceFloat32BeforeRound => "_publish"
    _ => raise @precision.PrecisionPlanError::IllegalPlacement
  }
  self.prefix + suffix
}
